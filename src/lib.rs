//! llama.cpp node family as a standalone Path 3 loadable plugin.
//!
//! Four streaming nodes wrapping the llama.cpp C library through safe
//! Rust bindings via the `llama-cpp-4` crate:
//!
//! - [`LlamaCppGenerationNode`] — Text generation (chat / completion)
//!   with token streaming, activation-tap side-channel, and inline
//!   `<tool_call>` parsing + dispatch (Qwen3 / Hermes / DeepSeek style).
//! - [`LlamaCppEmbeddingNode`] — Text → dense vector embeddings, with
//!   configurable pooling and L2 normalization.
//! - [`LlamaCppActivationNode`] — Capture hidden-state activations at
//!   arbitrary transformer layers via llama.cpp's `TensorCapture`
//!   callback.
//! - [`LlamaCppSteerNode`] — Inject activation deltas (emotion vectors,
//!   DoG) into generation. Currently runs in metadata mode (KV-cache
//!   injection pending — matches the in-tree implementation status).
//!
//! Originally lived in `remotemedia-core::nodes::llama_cpp` behind the
//! `llama-cpp` cargo feature; extracted here so the host crate doesn't
//! drag in `llama-cpp-sys-4` (~200 MiB CUDA-linked C build) just for
//! this node family.
//!
//! ## Node types exported
//!
//!   LlamaCppGenerationNode  — Text/Json → Text stream + Tensor (activation taps) + tool envelopes
//!   LlamaCppEmbeddingNode   — Text/Json → Tensor (dense embedding vector)
//!   LlamaCppActivationNode  — Text/Json → Tensor (one per requested layer)
//!   LlamaCppSteerNode       — Tensor / Text / Json → Text + Json (steering metadata)

mod llama_cpp;
mod tool_dispatch;
mod tool_spec;

use remotemedia_plugin_sdk::abi_stable::sabi_trait::TD_Opaque;
use remotemedia_plugin_sdk::abi_stable::std_types::{RErr, ROk, RResult, RString};
use remotemedia_plugin_sdk::adapter::StreamingNodeFfiAdapter;
use remotemedia_plugin_sdk::{FfiNodeBox, FfiNodeFactory, FfiNode_TO};
use serde_json::Value;

pub use crate::llama_cpp::{
    GpuOffload, LlamaBackendConfig, LlamaCppActivationConfig, LlamaCppActivationNode,
    LlamaCppActivationTapConfig, LlamaCppEmbeddingConfig, LlamaCppEmbeddingNode,
    LlamaCppGenerationConfig, LlamaCppGenerationNode, LlamaCppSteerConfig, LlamaCppSteerNode,
    LlamaCppSteerVector,
};

// ---------------------------------------------------------------------------
// Helpers — params → typed config
// ---------------------------------------------------------------------------

/// Parse the FFI `params` JSON string into a `serde_json::Value`. Empty
/// payload becomes `null` so individual `from_params` calls can apply
/// their own defaults via `unwrap_or_default()` semantics.
fn params_value(params: &RString) -> Result<Value, String> {
    let s = params.as_str();
    if s.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(s).map_err(|e| format!("invalid params JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Factories — one per node type
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct LlamaCppGenerationNodeFactory;

impl FfiNodeFactory for LlamaCppGenerationNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("LlamaCppGenerationNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let value = match params_value(&params) {
            Ok(v) => v,
            Err(e) => return RErr(RString::from(e)),
        };
        let node = match LlamaCppGenerationNode::from_params(&value) {
            Ok(n) => n,
            Err(e) => {
                return RErr(RString::from(format!(
                    "LlamaCppGenerationNode create failed: {e}"
                )));
            }
        };
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(node),
            TD_Opaque,
        ))
    }
}

#[derive(Default)]
pub struct LlamaCppEmbeddingNodeFactory;

impl FfiNodeFactory for LlamaCppEmbeddingNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("LlamaCppEmbeddingNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let value = match params_value(&params) {
            Ok(v) => v,
            Err(e) => return RErr(RString::from(e)),
        };
        let node = match LlamaCppEmbeddingNode::from_params(&value) {
            Ok(n) => n,
            Err(e) => {
                return RErr(RString::from(format!(
                    "LlamaCppEmbeddingNode create failed: {e}"
                )));
            }
        };
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(node),
            TD_Opaque,
        ))
    }
}

#[derive(Default)]
pub struct LlamaCppActivationNodeFactory;

impl FfiNodeFactory for LlamaCppActivationNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("LlamaCppActivationNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let value = match params_value(&params) {
            Ok(v) => v,
            Err(e) => return RErr(RString::from(e)),
        };
        let node = match LlamaCppActivationNode::from_params(&value) {
            Ok(n) => n,
            Err(e) => {
                return RErr(RString::from(format!(
                    "LlamaCppActivationNode create failed: {e}"
                )));
            }
        };
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(node),
            TD_Opaque,
        ))
    }
}

#[derive(Default)]
pub struct LlamaCppSteerNodeFactory;

impl FfiNodeFactory for LlamaCppSteerNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("LlamaCppSteerNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let value = match params_value(&params) {
            Ok(v) => v,
            Err(e) => return RErr(RString::from(e)),
        };
        let node = match LlamaCppSteerNode::from_params(&value) {
            Ok(n) => n,
            Err(e) => {
                return RErr(RString::from(format!(
                    "LlamaCppSteerNode create failed: {e}"
                )));
            }
        };
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(node),
            TD_Opaque,
        ))
    }
}

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

remotemedia_plugin_sdk::plugin_export!(
    LlamaCppGenerationNodeFactory,
    LlamaCppEmbeddingNodeFactory,
    LlamaCppActivationNodeFactory,
    LlamaCppSteerNodeFactory,
);
