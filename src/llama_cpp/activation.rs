//! LlamaCppActivationNode — hidden-state activation extraction via llama.cpp.
//!
//! Uses llama.cpp's `TensorCapture` callback to extract per-token hidden
//! states at arbitrary transformer layers during `llama_decode`.

use remotemedia_plugin_sdk::types::{Error, RuntimeData};
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::info;

use remotemedia_plugin_sdk::traits::runtime_context::InitializeContextRead;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;

use super::config::LlamaCppActivationConfig;
use super::inference::{self, ActivationCapture};

/// Llama.cpp activation extraction node.
pub struct LlamaCppActivationNode {
    config: LlamaCppActivationConfig,
    initialized: RwLock<bool>,
}

impl LlamaCppActivationNode {
    pub fn new(config: &LlamaCppActivationConfig) -> Result<Self, Error> {
        config
            .validate()
            .map_err(|e| Error::Execution(format!("Invalid config: {}", e)))?;

        Ok(Self {
            config: config.clone(),
            initialized: RwLock::new(false),
        })
    }

    pub fn from_params(params: &Value) -> Result<Self, Error> {
        let config: LlamaCppActivationConfig = serde_json::from_value(params.clone())
            .map_err(|e| Error::Execution(format!("Invalid config JSON: {}", e)))?;
        Self::new(&config)
    }

    async fn extract(&self, text: &str) -> Result<Vec<ActivationCapture>, Error> {
        let config = self.config.clone();
        let layers = config.layers.clone();
        let text = text.to_string();

        let result = tokio::task::spawn_blocking(move || {
            inference::run_activation(
                &config.model_path,
                &text,
                &layers,
                config.context_size,
                config.batch_size,
                config.backend.gpu_offload,
                config.backend.flash_attention,
                config.backend.threads,
                config.pooling,
            )
        })
        .await
        .map_err(|e| Error::Execution(format!("Task join failed: {}", e)))??;

        Ok(result)
    }

    fn l2_normalize(&self, vector: &mut Vec<f32>) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vector.iter_mut() {
                *v /= norm;
            }
        }
    }
}

#[async_trait::async_trait]
impl AsyncStreamingNode for LlamaCppActivationNode {
    fn node_type(&self) -> &str {
        "LlamaCppActivationNode"
    }

    async fn initialize(&self, ctx: &dyn InitializeContextRead) -> Result<(), Error> {
        info!(
            node = "llama-cpp-activation",
            model = %self.config.model_path,
            layers = ?self.config.layers,
            pooling = ?self.config.pooling,
            "Initializing LlamaCppActivationNode"
        );

        ctx.emit_progress(
            "loading_model",
            &format!(
                "Loading model for activation extraction: {} (layers: {:?})",
                self.config.model_path, self.config.layers
            ),
        );

        *self.initialized.write().await = true;
        ctx.emit_progress("ready", "LlamaCppActivationNode ready");
        Ok(())
    }

    async fn process(&self, data: RuntimeData) -> Result<RuntimeData, Error> {
        let text = match &data {
            RuntimeData::Text(text) => text.clone(),
            RuntimeData::Json(value) => value
                .get("text")
                .or(value.get("prompt"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| value.to_string()),
            other => {
                return Err(Error::Execution(format!(
                    "LlamaCppActivationNode accepts Text or Json, got {}",
                    other.data_type()
                )));
            }
        };

        let emotion_label = match &data {
            RuntimeData::Json(value) => value
                .get("emotion")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            _ => None,
        };

        let captures = self.extract(&text).await?;

        let capture = captures
            .first()
            .ok_or_else(|| Error::Execution("No activations captured".to_string()))?;

        let mut vector = capture.activation.clone();
        if self.config.normalize {
            self.l2_normalize(&mut vector);
        }

        let tensor_data: Vec<u8> = vector.iter().flat_map(|&x| x.to_le_bytes()).collect();

        let mut metadata = serde_json::json!({
            "model": self.config.model_path,
            "layer": capture.layer,
            "hidden_size": capture.hidden_size,
            "pooling": format!("{:?}", self.config.pooling),
            "normalized": self.config.normalize,
            "raw_norm": capture.raw_norm,
        });

        if let Some(emotion) = &emotion_label {
            metadata["emotion"] = serde_json::json!(emotion);
        }

        Ok(RuntimeData::Tensor {
            data: tensor_data,
            shape: vec![vector.len() as i32],
            dtype: 0,
            metadata: Some(metadata),
        })
    }

    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        _session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let text = match &data {
            RuntimeData::Text(text) => text.clone(),
            RuntimeData::Json(value) => value
                .get("text")
                .or(value.get("prompt"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| value.to_string()),
            other => {
                return Err(Error::Execution(format!(
                    "LlamaCppActivationNode accepts Text or Json, got {}",
                    other.data_type()
                )));
            }
        };

        let emotion_label = match &data {
            RuntimeData::Json(value) => value
                .get("emotion")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            _ => None,
        };

        let captures = self.extract(&text).await?;
        let mut count = 0;

        for capture in captures {
            let mut vector = capture.activation.clone();
            if self.config.normalize {
                self.l2_normalize(&mut vector);
            }

            let tensor_data: Vec<u8> = vector.iter().flat_map(|&x| x.to_le_bytes()).collect();

            let mut metadata = serde_json::json!({
                "model": self.config.model_path,
                "layer": capture.layer,
                "hidden_size": capture.hidden_size,
                "pooling": format!("{:?}", self.config.pooling),
                "normalized": self.config.normalize,
                "raw_norm": capture.raw_norm,
            });

            if let Some(emotion) = &emotion_label {
                metadata["emotion"] = serde_json::json!(emotion);
            }

            callback(RuntimeData::Tensor {
                data: tensor_data,
                shape: vec![vector.len() as i32],
                dtype: 0,
                metadata: Some(metadata),
            })?;
            count += 1;
        }

        Ok(count)
    }

    async fn process_control_message(
        &self,
        _message: RuntimeData,
        _session_id: Option<String>,
    ) -> Result<bool, Error> {
        Ok(false)
    }
}
