//! Configuration types for llama.cpp nodes.
//!
//! Plugin-side port: `schemars::JsonSchema` derives dropped because the
//! plugin doesn't surface schemas at the FFI boundary. `crate::nodes::tool_spec`
//! → `crate::tool_spec` (inlined).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool_spec::ToolSpec;

// ---------------------------------------------------------------------------
// Shared backend config
// ---------------------------------------------------------------------------

/// GPU offload strategy for the llama.cpp backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuOffload {
    /// Run entirely on CPU.
    None,
    /// Offload all possible layers to GPU.
    All,
    /// Offload a specific number of layers (0 = CPU only).
    Layers(u16),
}

impl Default for GpuOffload {
    fn default() -> Self {
        Self::None
    }
}

/// llama.cpp backend initialization settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaBackendConfig {
    /// Whether to enable NUMA-aware memory allocation.
    pub numa: bool,
    /// GPU offload strategy.
    pub gpu_offload: GpuOffload,
    /// Flash Attention 2 (reduces memory, may change outputs slightly).
    pub flash_attention: bool,
    /// Number of threads for computation. `0` = auto (all cores).
    pub threads: Option<u32>,
    /// Number of threads for the background I/O thread. `0` = auto.
    pub threads_batch: Option<u32>,
}

impl Default for LlamaBackendConfig {
    fn default() -> Self {
        Self {
            numa: false,
            gpu_offload: GpuOffload::default(),
            flash_attention: false,
            threads: None,
            threads_batch: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Generation config
// ---------------------------------------------------------------------------

/// Configuration for `LlamaCppGenerationNode`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaCppGenerationConfig {
    /// Path to a GGUF model file (local path or HuggingFace-style repo).
    pub model_path: String,
    /// Backend settings.
    pub backend: LlamaBackendConfig,
    /// Context size (max tokens the model can attend to).
    #[serde(alias = "n_ctx")]
    pub context_size: u32,
    /// Batch size for decoding.
    #[serde(alias = "n_batch")]
    pub batch_size: u32,
    /// Maximum tokens to generate.
    #[serde(alias = "max_tokens")]
    pub max_tokens: u32,
    /// Sampling temperature (0.0 = greedy, 1.0 = max randomness).
    pub temperature: f32,
    /// Top-p nucleus sampling cutoff.
    #[serde(alias = "top_p")]
    pub top_p: f32,
    /// Top-k sampling cutoff (0 = disabled).
    #[serde(alias = "top_k")]
    pub top_k: u32,
    /// Min-p sampling cutoff (0.0 = disabled).
    #[serde(alias = "min_p")]
    pub min_p: f32,
    /// Repeat penalty (1.0 = disabled).
    #[serde(alias = "repeat_penalty")]
    pub repeat_penalty: f32,
    /// System prompt prepended to every generation request.
    #[serde(alias = "system_prompt")]
    pub system_prompt: Option<String>,
    /// Random seed for sampling (0 = random seed each time).
    pub seed: u64,
    /// Optional activation-tap configuration.
    #[serde(default)]
    pub activation_tap: Option<LlamaCppActivationTapConfig>,

    /// Additional user-defined tools. The built-in `say` and `show`
    /// tools are always advertised to the model.
    #[serde(default, alias = "tools")]
    pub tools: Vec<ToolSpec>,

    /// If `Some`, only tools with names in this list are advertised to
    /// the model.
    #[serde(default)]
    pub active_tools: Option<Vec<String>>,

    /// Forwarded as the `tool_choice` Jinja kwarg.
    #[serde(default)]
    pub tool_choice: Option<Value>,
}

/// Configuration for the activation-tap side-channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaCppActivationTapConfig {
    /// Layer index whose `l_out-{layer}` output tensor is captured.
    pub layer: u32,
    /// Cadence: capture and emit one envelope every N generated
    /// response tokens. `0` = input-only mode.
    pub every_n_tokens: u32,
}

impl Default for LlamaCppActivationTapConfig {
    fn default() -> Self {
        Self {
            layer: 15,
            every_n_tokens: 32,
        }
    }
}

impl Default for LlamaCppGenerationConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            backend: LlamaBackendConfig::default(),
            context_size: 4096,
            batch_size: 512,
            max_tokens: 256,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.0,
            repeat_penalty: 1.1,
            system_prompt: None,
            seed: 0,
            activation_tap: None,
            tools: Vec::new(),
            active_tools: None,
            tool_choice: None,
        }
    }
}

impl LlamaCppGenerationConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.model_path.is_empty() {
            return Err("model_path must not be empty".to_string());
        }
        if self.context_size == 0 {
            return Err("context_size must be > 0".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be > 0".to_string());
        }
        if self.temperature < 0.0 {
            return Err("temperature must be >= 0".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Embedding config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaCppEmbeddingConfig {
    pub model_path: String,
    pub backend: LlamaBackendConfig,
    #[serde(alias = "n_ctx")]
    pub context_size: u32,
    #[serde(alias = "n_batch")]
    pub batch_size: u32,
    pub pooling: EmbeddingPooling,
    #[serde(alias = "normalize")]
    pub l2_normalize: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingPooling {
    Mean,
    LastToken,
    FirstToken,
    Cls,
}

impl Default for EmbeddingPooling {
    fn default() -> Self {
        Self::Mean
    }
}

impl Default for LlamaCppEmbeddingConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            backend: LlamaBackendConfig::default(),
            context_size: 512,
            batch_size: 256,
            pooling: EmbeddingPooling::Mean,
            l2_normalize: true,
        }
    }
}

impl LlamaCppEmbeddingConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_path.is_empty() {
            return Err("model_path must not be empty".to_string());
        }
        if self.context_size == 0 {
            return Err("context_size must be > 0".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Activation config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaCppActivationConfig {
    pub model_path: String,
    pub backend: LlamaBackendConfig,
    #[serde(alias = "n_ctx")]
    pub context_size: u32,
    #[serde(alias = "n_batch")]
    pub batch_size: u32,
    pub layers: Vec<usize>,
    pub pooling: ActivationPooling,
    pub normalize: bool,
    #[serde(alias = "system_prompt")]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationPooling {
    Mean,
    LastToken,
    FirstToken,
}

impl Default for ActivationPooling {
    fn default() -> Self {
        Self::LastToken
    }
}

impl Default for LlamaCppActivationConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            backend: LlamaBackendConfig::default(),
            context_size: 4096,
            batch_size: 512,
            layers: vec![21],
            pooling: ActivationPooling::LastToken,
            normalize: true,
            system_prompt: None,
        }
    }
}

impl LlamaCppActivationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_path.is_empty() {
            return Err("model_path must not be empty".to_string());
        }
        if self.layers.is_empty() {
            return Err("layers must contain at least one layer index".to_string());
        }
        if self.context_size == 0 {
            return Err("context_size must be > 0".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Steering config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppSteerVector {
    pub label: String,
    #[serde(default = "default_zero")]
    pub coefficient: f32,
}

fn default_zero() -> f32 {
    0.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaCppSteerConfig {
    pub model_path: String,
    pub backend: LlamaBackendConfig,
    #[serde(alias = "n_ctx")]
    pub context_size: u32,
    #[serde(alias = "n_batch")]
    pub batch_size: u32,
    pub layer: usize,
    #[serde(alias = "layer_norm")]
    pub layer_norm_value: f32,
    pub vectors: Vec<LlamaCppSteerVector>,
    #[serde(alias = "max_coefficient")]
    pub max_coefficient: f32,
    pub generation: LlamaCppGenerationConfig,
    #[serde(alias = "system_prompt")]
    pub system_prompt: Option<String>,
}

impl Default for LlamaCppSteerConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            backend: LlamaBackendConfig::default(),
            context_size: 4096,
            batch_size: 512,
            layer: 21,
            layer_norm_value: 14.7,
            vectors: Vec::new(),
            max_coefficient: 1.0,
            generation: LlamaCppGenerationConfig::default(),
            system_prompt: None,
        }
    }
}

impl LlamaCppSteerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_path.is_empty() {
            return Err("model_path must not be empty".to_string());
        }
        if self.vectors.is_empty() {
            return Err("at least one steering vector must be configured".to_string());
        }
        if self.layer_norm_value <= 0.0 {
            return Err("layer_norm_value must be > 0".to_string());
        }
        if self.max_coefficient <= 0.0 {
            return Err("max_coefficient must be > 0".to_string());
        }
        Ok(())
    }
}
