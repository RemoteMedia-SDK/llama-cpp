//! LlamaCppEmbeddingNode — text embeddings via llama.cpp.
//!
//! Accepts `RuntimeData::Text` and emits `RuntimeData::Tensor`
//! containing the dense embedding vector.
//!
//! Runs inference on a blocking thread (llama.cpp types are not Send).

use remotemedia_plugin_sdk::types::{Error, RuntimeData};
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::info;

use remotemedia_plugin_sdk::traits::runtime_context::InitializeContextRead;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;

use super::config::LlamaCppEmbeddingConfig;
use super::inference;

/// Llama.cpp embedding node.
pub struct LlamaCppEmbeddingNode {
    config: LlamaCppEmbeddingConfig,
    initialized: RwLock<bool>,
}

impl LlamaCppEmbeddingNode {
    pub fn new(config: &LlamaCppEmbeddingConfig) -> Result<Self, Error> {
        config
            .validate()
            .map_err(|e| Error::Execution(format!("Invalid config: {}", e)))?;

        Ok(Self {
            config: config.clone(),
            initialized: RwLock::new(false),
        })
    }

    pub fn from_params(params: &Value) -> Result<Self, Error> {
        let config: LlamaCppEmbeddingConfig = serde_json::from_value(params.clone())
            .map_err(|e| Error::Execution(format!("Invalid config JSON: {}", e)))?;
        Self::new(&config)
    }

    async fn embed(&self, text: &str) -> Result<(Vec<f32>, usize), Error> {
        let config = self.config.clone();
        let text = text.to_string();

        let result = tokio::task::spawn_blocking(move || {
            inference::run_embedding(
                &config.model_path,
                &text,
                config.context_size,
                config.batch_size,
                config.backend.gpu_offload,
                config.backend.flash_attention,
                config.backend.threads,
            )
        })
        .await
        .map_err(|e| Error::Execution(format!("Task join failed: {}", e)))??;

        Ok((result.embedding, result.hidden_size))
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
impl AsyncStreamingNode for LlamaCppEmbeddingNode {
    fn node_type(&self) -> &str {
        "LlamaCppEmbeddingNode"
    }

    async fn initialize(&self, ctx: &dyn InitializeContextRead) -> Result<(), Error> {
        info!(
            node = "llama-cpp-embedding",
            model = %self.config.model_path,
            "Initializing LlamaCppEmbeddingNode"
        );

        ctx.emit_progress(
            "loading_model",
            &format!("Loading embedding model: {}", self.config.model_path),
        );

        *self.initialized.write().await = true;
        ctx.emit_progress("ready", "LlamaCppEmbeddingNode ready");
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
                    "LlamaCppEmbeddingNode accepts Text or Json, got {}",
                    other.data_type()
                )));
            }
        };

        let (mut embedding, hidden_size) = self.embed(&text).await?;

        if self.config.l2_normalize {
            self.l2_normalize(&mut embedding);
        }

        let tensor_data: Vec<u8> = embedding.iter().flat_map(|&x| x.to_le_bytes()).collect();

        Ok(RuntimeData::Tensor {
            data: tensor_data,
            shape: vec![hidden_size as i32],
            dtype: 0, // float32
            metadata: Some(serde_json::json!({
                "model": self.config.model_path,
                "pooling": format!("{:?}", self.config.pooling),
                "normalized": self.config.l2_normalize,
            })),
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
        let output = self.process(data).await?;
        callback(output)?;
        Ok(1)
    }

    async fn process_control_message(
        &self,
        _message: RuntimeData,
        _session_id: Option<String>,
    ) -> Result<bool, Error> {
        Ok(false)
    }
}
