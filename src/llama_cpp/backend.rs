//! Process-wide singleton for `llama_cpp_4::LlamaBackend`.
//!
//! `LlamaBackend::init()` flips a global `AtomicBool` inside the binding and
//! returns `BackendAlreadyInitialized` on every subsequent call. Each
//! `LlamaCppGenerationNode` worker (plus the one-shot helpers in `inference.rs`)
//! used to call it directly, which meant the second pipeline session in the
//! same process always failed to start. We init once here and hand out a
//! `&'static LlamaBackend` to every caller.

use std::sync::{Mutex, OnceLock};

use llama_cpp_4::llama_backend::LlamaBackend;
use remotemedia_plugin_sdk::types::Error;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Lazily initialize the process-global llama.cpp backend and return a
/// reference. Safe to call from any thread; first caller wins, others
/// receive the same `&'static LlamaBackend`. We serialize through a
/// dedicated mutex (rather than `OnceLock::get_or_init` which can't return
/// `Result`) so a race-loser never constructs a duplicate `LlamaBackend`
/// whose `Drop` would call `llama_backend_free()` on the global.
pub(crate) fn get_or_init() -> Result<&'static LlamaBackend, Error> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let _guard = INIT_LOCK
        .lock()
        .map_err(|_| Error::Execution("LlamaBackend init mutex poisoned".into()))?;
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let backend = LlamaBackend::init()
        .map_err(|e| Error::Execution(format!("LlamaBackend::init failed: {}", e)))?;
    let _ = BACKEND.set(backend);
    Ok(BACKEND.get().expect("just stored"))
}
