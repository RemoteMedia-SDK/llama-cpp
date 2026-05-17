//! llama.cpp node family — native GGUF inference via [llama-cpp-4](https://crates.io/crates/llama-cpp-4).
//!
//! Four streaming nodes wrapping the llama.cpp C library through safe
//! Rust bindings:
//!
//! | Node | Purpose |
//! |---|---|
//! | [`LlamaCppGenerationNode`] | Text generation (chat / completion) w/ token streaming + tool calls |
//! | [`LlamaCppEmbeddingNode`] | Text → dense vector embeddings |
//! | [`LlamaCppActivationNode`] | Capture hidden-state activations at arbitrary layers |
//! | [`LlamaCppSteerNode`] | Inject activation deltas (emotion steering, DoG) |
//!
//! This is the Path-3 plugin extraction of the in-tree
//! `remotemedia-core::nodes::llama_cpp` module. Behaviour is preserved
//! verbatim; the only structural differences are:
//!
//! - the `factory.rs` `NodeProvider` registration is replaced with
//!   hand-rolled `FfiNodeFactory` impls in the parent `lib.rs`
//! - the `StreamingNodeFactory` / `StreamingNode` wrappers are dropped
//!   (the FFI adapter operates directly on `AsyncStreamingNode`)
//! - the `NodeSchema` / `MediaCapabilities` declarations are dropped
//!   (not exposed at the FFI boundary)
//! - the `projection.rs` (`activation-face` feature, calibration tooling
//!   for downstream face nodes) is left in the host crate
//! - the `ChatState` per-session bookkeeping is collapsed onto a fresh
//!   per-call state, because `make_session_state` / `try_state` is a
//!   host-only mechanism not reachable from the FFI surface. Multi-turn
//!   coordination must therefore happen outside the plugin (consumer
//!   side) for now.

pub mod activation;
pub mod backend;
pub mod config;
pub mod embedding;
pub mod generation;
pub mod inference;
pub mod steer;

pub use activation::LlamaCppActivationNode;
pub use config::{
    GpuOffload, LlamaBackendConfig, LlamaCppActivationConfig, LlamaCppActivationTapConfig,
    LlamaCppEmbeddingConfig, LlamaCppGenerationConfig, LlamaCppSteerConfig, LlamaCppSteerVector,
};
pub use embedding::LlamaCppEmbeddingNode;
pub use generation::LlamaCppGenerationNode;
pub use steer::LlamaCppSteerNode;
