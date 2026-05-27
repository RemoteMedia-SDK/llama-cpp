//! LlamaCppGenerationNode — text generation via llama.cpp.
//!
//! Accepts `RuntimeData::Text` (user prompt) or `RuntimeData::Json`
//! (structured messages) and streams generated tokens downstream.
//!
//! # Architecture
//!
//! llama.cpp types (`LlamaBackend`, `LlamaModel`, `LlamaContext`,
//! `LlamaBatch`) contain raw C pointers and are not `Send`. They cannot
//! cross tokio task boundaries.
//!
//! To get a single load + reuse on every turn, this node spawns a
//! dedicated `std::thread` during `initialize()`. That thread:
//!   1. Initializes the llama backend (registers ggml-cuda etc.)
//!   2. Loads the GGUF model with the configured GPU-offload setting
//!   3. Creates a long-lived `LlamaContext`
//!   4. Sits in a request loop, decoding prompts as they arrive
//!
//! # Plugin-side differences from the in-tree module
//!
//! - **Per-session `ChatState`** — the in-tree node uses the host's
//!   `make_session_state` / `try_state` machinery to track per-session
//!   conversation history across calls. That surface isn't reachable
//!   from the Path-3 FFI boundary; this plugin therefore constructs a
//!   fresh `ChatState` per `generate()` call. Multi-turn coordination
//!   must happen outside the plugin (e.g. the consumer batches
//!   history into the prompt itself).
//! - **`CancelGate`** — the in-tree dispatcher takes a `ProtectGuard`
//!   off the per-(session, node) `CancelGate` whenever a
//!   `cancelable: false` tool is dispatched, so subsequent `barge_in`
//!   envelopes are suppressed for the rest of the turn. The plugin
//!   doesn't have access to `CancelGate` (it's a host-internal type),
//!   so the dispatcher just calls the tool dispatcher directly and
//!   relies on the universal future-drop cancellation pathway.

use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;

use remotemedia_plugin_sdk::types::{ControlMessageType, Error, RuntimeData, TEXT_CHANNEL_DEFAULT};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use remotemedia_plugin_sdk::traits::runtime_context::InitializeContextRead;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;
use remotemedia_plugin_sdk::traits::{InterruptableBackend, StatefulConversationBackend};

use super::config::LlamaCppGenerationConfig;
use crate::tool_spec::{default_say_tool, default_show_tool, ToolSpec};

/// One hidden-state snapshot captured by the activation tap.
#[derive(Debug, Clone)]
pub(crate) struct TappedActivation {
    pub layer: u32,
    pub hidden: Vec<f32>,
    pub phase: &'static str,
    pub token_index: u32,
    pub turn_offset_ms: u64,
}

/// One live event emitted by the worker during a chat turn.
#[derive(Debug)]
pub(crate) enum TurnEvent {
    Chunk(String),
    Tap(TappedActivation),
    Error(Error),
}

/// Convert a tapped activation into a `RuntimeData::Tensor` envelope.
fn tap_to_runtime_data(tap: TappedActivation) -> RuntimeData {
    use bytemuck::cast_slice;
    let n_embd = tap.hidden.len();
    let bytes: Vec<u8> = cast_slice::<f32, u8>(&tap.hidden).to_vec();
    let metadata = serde_json::json!({
        "kind": "activation_tap",
        "layer": tap.layer,
        "phase": tap.phase,
        "token_index": tap.token_index,
        "turn_offset_ms": tap.turn_offset_ms,
    });
    RuntimeData::Tensor {
        data: bytes,
        shape: vec![n_embd as i32],
        dtype: 0,
        metadata: Some(metadata),
    }
}

enum WorkerRequest {
    Generate {
        session_id: String,
        chat: Arc<Mutex<ChatState>>,
        prompt: String,
        tools_json: Option<Value>,
        tool_choice: Option<Value>,
        event_tx: mpsc::Sender<TurnEvent>,
        cancel_flag: Arc<AtomicBool>,
    },
    #[allow(dead_code)]
    ResetSession {
        session_id: String,
        chat: Arc<Mutex<ChatState>>,
    },
}

/// Llama.cpp text generation node.
pub struct LlamaCppGenerationNode {
    config: LlamaCppGenerationConfig,
    worker_tx: OnceLock<mpsc::Sender<WorkerRequest>>,
    sessions: Arc<Mutex<HashMap<String, Arc<Mutex<ChatState>>>>>,
    interrupts: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

impl LlamaCppGenerationNode {
    pub fn new(config: &LlamaCppGenerationConfig) -> Result<Self, Error> {
        config
            .validate()
            .map_err(|e| Error::Execution(format!("Invalid config: {}", e)))?;

        Ok(Self {
            config: config.clone(),
            worker_tx: OnceLock::new(),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            interrupts: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn from_params(params: &Value) -> Result<Self, Error> {
        let config: LlamaCppGenerationConfig = serde_json::from_value(params.clone())
            .map_err(|e| Error::Execution(format!("Invalid config JSON: {}", e)))?;
        Self::new(&config)
    }

    async fn generate_for_session(
        &self,
        session_id: &str,
        chat: Arc<Mutex<ChatState>>,
        prompt: &str,
        tools_json: Option<Value>,
        tool_choice: Option<Value>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<mpsc::Receiver<TurnEvent>, Error> {
        let tx = self.worker_tx.get().ok_or_else(|| {
            Error::Execution(
                "LlamaCppGenerationNode worker not running — initialize() was \
                 not called or model load failed"
                    .into(),
            )
        })?;

        let (event_tx, event_rx) = mpsc::channel::<TurnEvent>(64);
        tx.send(WorkerRequest::Generate {
            session_id: session_id.to_string(),
            chat,
            prompt: prompt.to_string(),
            tools_json,
            tool_choice,
            event_tx,
            cancel_flag,
        })
        .await
        .map_err(|_| Error::Execution("LlamaCpp worker thread is gone".into()))?;

        Ok(event_rx)
    }

    pub(crate) fn fresh_chat_state(&self) -> ChatState {
        let mut chat = ChatState::new();
        if let Some(sys) = self.config.system_prompt.as_ref().filter(|s| !s.is_empty()) {
            chat.messages.push(ChatMsg {
                role: "system".to_string(),
                content: sys.clone(),
            });
        }
        chat
    }

    pub(crate) fn build_tool_registry(&self) -> Vec<ToolSpec> {
        let mut out: Vec<ToolSpec> = Vec::new();
        out.push(default_say_tool());
        out.push(default_show_tool());
        for spec in &self.config.tools {
            if let Some(existing) = out.iter_mut().find(|t| t.name == spec.name) {
                *existing = spec.clone();
            } else {
                out.push(spec.clone());
            }
        }
        if let Some(active) = &self.config.active_tools {
            out.retain(|t| active.iter().any(|n| n == &t.name));
        }
        out
    }

    pub(crate) fn resolved_tools_json(&self) -> Option<Value> {
        let registry = self.build_tool_registry();
        if registry.is_empty() {
            None
        } else {
            Some(crate::tool_spec::to_openai_tools_array(&registry))
        }
    }
}

#[async_trait::async_trait]
impl AsyncStreamingNode for LlamaCppGenerationNode {
    fn node_type(&self) -> &str {
        "LlamaCppGenerationNode"
    }

    async fn initialize(&self, ctx: &dyn InitializeContextRead) -> Result<(), Error> {
        info!(
            model = %self.config.model_path,
            context_size = self.config.context_size,
            "Initializing LlamaCppGenerationNode"
        );

        ctx.emit_progress(
            "loading_model",
            &format!("Loading model: {}", self.config.model_path),
        );

        let (req_tx, req_rx) = mpsc::channel::<WorkerRequest>(8);
        let (init_tx, init_rx) = oneshot::channel::<Result<(), Error>>();

        let config = self.config.clone();
        std::thread::Builder::new()
            .name("llama-cpp-gen".to_string())
            .spawn(move || worker_main(config, req_rx, init_tx))
            .map_err(|e| {
                Error::Execution(format!("Failed to spawn llama.cpp worker thread: {}", e))
            })?;

        init_rx.await.map_err(|_| {
            Error::Execution("llama.cpp worker exited before reporting init result".into())
        })??;

        if self.worker_tx.set(req_tx).is_err() {
            return Err(Error::Execution(
                "LlamaCppGenerationNode worker channel already set \
                 (initialize() called twice?)"
                    .into(),
            ));
        }

        ctx.emit_progress("ready", "LlamaCppGenerationNode ready");
        Ok(())
    }

    async fn process(&self, data: RuntimeData) -> Result<RuntimeData, Error> {
        let prompt = match &data {
            RuntimeData::Text(text) => text.clone(),
            RuntimeData::Json(value) => value
                .get("prompt")
                .or(value.get("text"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| value.to_string()),
            other => {
                return Err(Error::Execution(format!(
                    "LlamaCppGenerationNode accepts Text or Json, got {}",
                    other.data_type()
                )));
            }
        };

        let chat = Arc::new(Mutex::new(self.fresh_chat_state()));
        let tools_json = self.resolved_tools_json();
        let tool_choice = self.config.tool_choice.clone();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let mut events = self
            .generate_for_session("__inner_oneshot__", chat, &prompt, tools_json, tool_choice, cancel_flag)
            .await?;
        let mut joined = String::new();
        while let Some(event) = events.recv().await {
            match event {
                TurnEvent::Chunk(s) => joined.push_str(&s),
                TurnEvent::Tap(_) => {}
                TurnEvent::Error(e) => return Err(e),
            }
        }
        Ok(RuntimeData::Text(joined))
    }

    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let prompt = match &data {
            RuntimeData::Text(text) => text.clone(),
            RuntimeData::Json(value) => value
                .get("prompt")
                .or(value.get("text"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| value.to_string()),
            other => {
                return Err(Error::Execution(format!(
                    "LlamaCppGenerationNode accepts Text or Json, got {}",
                    other.data_type()
                )));
            }
        };

        // Persistent session: look up or create ChatState from the sessions map.
        let session_key = session_id.as_deref().unwrap_or("__inner_oneshot__");
        let chat = {
            let mut sessions = self.sessions.lock().unwrap();
            sessions.entry(session_key.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(self.fresh_chat_state())))
                .clone()
        };

        // Reset interrupt flag and get cancel latch for this turn.
        let cancel_flag = {
            let mut interrupts = self.interrupts.lock().unwrap();
            let flag = interrupts.entry(session_key.to_string())
                .or_insert_with(|| Arc::new(AtomicBool::new(false)));
            flag.store(false, Ordering::Relaxed);
            flag.clone()
        };

        let turn_started = std::time::Instant::now();

        let tools_json = self.resolved_tools_json();
        let tool_choice = self.config.tool_choice.clone();
        let events = self
            .generate_for_session(
                session_key,
                chat.clone(),
                &prompt,
                tools_json,
                tool_choice,
                cancel_flag.clone(),
            )
            .await?;
        let registry = self.build_tool_registry();
        let count = stream_events_with_tool_dispatch(
            events,
            &registry,
            TEXT_CHANNEL_DEFAULT,
            &mut callback,
        )
        .await?;

        // Handle early-interrupt history rollback inline.
        let interrupted = cancel_flag.load(Ordering::Relaxed);
        let elapsed = turn_started.elapsed().as_secs_f32();
        if interrupted && elapsed < 3.0 {
            if let Some(chat_entry) = self.sessions.lock().unwrap().get(session_key) {
                let mut guard = chat_entry.lock().unwrap();
                if guard.messages.len() >= 2 {
                    guard.messages.pop(); // incomplete assistant reply
                    if guard.messages.last().map(|m| m.role == "user").unwrap_or(false) {
                        guard.messages.pop(); // user prompt that triggered it
                    }
                }
            }
        }

        Ok(count)
    }

    async fn process_control_message(
        &self,
        message: RuntimeData,
        _session_id: Option<String>,
    ) -> Result<bool, Error> {
        if let RuntimeData::ControlMessage { message_type, .. } = &message {
            if let ControlMessageType::CancelSpeculation { .. } = message_type {
                debug!("Received cancel speculation message for LlamaCpp generation");
                return Ok(true);
            }
        }
        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// Worker thread (`!Send` llama.cpp objects live here)
// ---------------------------------------------------------------------------

fn install_llama_log_filter() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        unsafe extern "C" fn filter(
            level: llama_cpp_sys_4::ggml_log_level,
            text: *const std::os::raw::c_char,
            _user_data: *mut std::os::raw::c_void,
        ) {
            if level >= llama_cpp_sys_4::GGML_LOG_LEVEL_WARN && !text.is_null() {
                let s = std::ffi::CStr::from_ptr(text).to_string_lossy();
                eprint!("{}", s);
            }
        }
        unsafe {
            llama_cpp_sys_4::llama_log_set(Some(filter), std::ptr::null_mut());
        }
    });
}

fn worker_main(
    config: LlamaCppGenerationConfig,
    mut req_rx: mpsc::Receiver<WorkerRequest>,
    init_tx: oneshot::Sender<Result<(), Error>>,
) {
    use super::config::GpuOffload;
    use llama_cpp_4::model::params::LlamaModelParams;
    use llama_cpp_4::model::LlamaModel;
    use std::time::Instant;

    let started = Instant::now();
    info!("llama.cpp worker: initializing backend");
    let backend = match super::backend::get_or_init() {
        Ok(b) => b,
        Err(e) => {
            error!("{}", e);
            let _ = init_tx.send(Err(e));
            return;
        }
    };

    install_llama_log_filter();

    let n_gpu_layers = match config.backend.gpu_offload {
        GpuOffload::None => 0,
        GpuOffload::All => 1000,
        GpuOffload::Layers(n) => n as u32,
    };
    info!(
        model = %config.model_path,
        n_gpu_layers,
        "llama.cpp worker: loading model (this may take 30-60 s for large GGUF files)"
    );
    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

    let model = match LlamaModel::load_from_file(backend, &config.model_path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            let err = Error::Execution(format!("Model load failed: {}", e));
            error!("{}", err);
            let _ = init_tx.send(Err(err));
            return;
        }
    };
    info!(
        load_ms = started.elapsed().as_millis() as u64,
        "llama.cpp worker: model loaded"
    );

    let ctx_started = Instant::now();
    info!(
        n_ctx = config.context_size,
        n_batch = config.batch_size,
        flash_attention = config.backend.flash_attention,
        "llama.cpp worker: creating context"
    );

    let mut ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    ctx_params = ctx_params.with_n_ctx(std::num::NonZeroU32::new(config.context_size));
    ctx_params = ctx_params.with_n_batch(config.batch_size);
    if config.backend.flash_attention {
        ctx_params = ctx_params.with_flash_attention(true);
    }
    if let Some(threads) = config.backend.threads {
        ctx_params = ctx_params.with_n_threads(threads as i32);
    }

    let mut tap_capture = config.activation_tap.as_ref().map(|t| {
        llama_cpp_4::context::tensor_capture::TensorCapture::for_layers(&[t.layer as usize])
    });
    if let Some(cap) = tap_capture.as_mut() {
        ctx_params = ctx_params.with_tensor_capture(cap);
    }

    let mut llama_ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            let err = Error::Execution(format!("Context creation failed: {}", e));
            error!("{}", err);
            let _ = init_tx.send(Err(err));
            return;
        }
    };
    info!(
        ctx_ms = ctx_started.elapsed().as_millis() as u64,
        total_ms = started.elapsed().as_millis() as u64,
        "llama.cpp worker: ready for inference"
    );

    let template = ChatTemplate::from_model(&model);
    info!(
        jinja_template = template.has_template,
        "llama.cpp worker: chat template renderer initialized"
    );

    if init_tx.send(Ok(())).is_err() {
        return;
    }

    while let Some(req) = req_rx.blocking_recv() {
        match req {
            WorkerRequest::Generate {
                session_id: req_session,
                chat,
                prompt,
                tools_json,
                tool_choice,
                event_tx,
                cancel_flag,
            } => {
                let t0 = Instant::now();
                let mut chat_guard = match chat.lock() {
                    Ok(g) => g,
                    Err(poisoned) => {
                        warn!(
                            session = %req_session,
                            "llama.cpp worker: session ChatState lock was poisoned; recovering"
                        );
                        poisoned.into_inner()
                    }
                };
                debug!(
                    session = %req_session,
                    prompt_len = prompt.len(),
                    history_messages = chat_guard.messages.len(),
                    "llama.cpp worker: generation request"
                );
                let result = run_turn_incremental(
                    &mut *chat_guard,
                    &template,
                    &model,
                    &mut llama_ctx,
                    &config,
                    &prompt,
                    tools_json.as_ref(),
                    tool_choice.as_ref(),
                    tap_capture.as_mut(),
                    &event_tx,
                    cancel_flag.clone(),
                );
                match &result {
                    Ok(stats) => {
                        info!(
                            session = %req_session,
                            n_decoded = stats.n_decoded,
                            n_reused = stats.n_reused,
                            n_response_tokens = stats.n_response_tokens,
                            elapsed_ms = t0.elapsed().as_millis() as u64,
                            "llama.cpp worker: generation complete"
                        );
                    }
                    Err(e) => {
                        error!(
                            session = %req_session,
                            "llama.cpp worker: generation failed: {}", e
                        );
                        let _ = event_tx
                            .blocking_send(TurnEvent::Error(Error::Execution(e.to_string())));
                    }
                }
                drop(chat_guard);
                drop(event_tx);
            }
            WorkerRequest::ResetSession {
                session_id: req_session,
                chat,
            } => {
                let mut chat_guard = chat.lock().unwrap_or_else(|p| p.into_inner());
                let kept_system = chat_guard.clear_keep_system();
                info!(
                    session = %req_session,
                    kept_system,
                    "llama.cpp worker: session reset"
                );
            }
        }
    }

    info!("llama.cpp worker: channel closed, shutting down");
    drop(llama_ctx);
    drop(model);
}

/// Conversation message.
#[derive(Clone)]
pub(crate) struct ChatMsg {
    pub(crate) role: String,
    pub(crate) content: String,
}

pub struct ChatState {
    pub(crate) messages: Vec<ChatMsg>,
}

impl ChatState {
    pub(crate) fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub(crate) fn clear_keep_system(&mut self) -> bool {
        let system = self
            .messages
            .first()
            .filter(|m| m.role == "system")
            .cloned();
        self.messages.clear();
        if let Some(sys) = system {
            self.messages.push(sys);
            true
        } else {
            false
        }
    }
}

fn to_llama_messages(msgs: &[ChatMsg]) -> Result<Vec<llama_cpp_4::model::LlamaChatMessage>, Error> {
    msgs.iter()
        .map(|m| {
            llama_cpp_4::model::LlamaChatMessage::new(m.role.clone(), m.content.clone())
                .map_err(|e| Error::Execution(format!("invalid chat message: {}", e)))
        })
        .collect()
}

/// Renderer for the GGUF model's embedded Jinja chat template.
struct ChatTemplate {
    env: minijinja::Environment<'static>,
    has_template: bool,
}

impl ChatTemplate {
    fn from_model(model: &llama_cpp_4::model::LlamaModel) -> Self {
        let mut env = minijinja::Environment::new();

        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

        env.add_function(
            "raise_exception",
            |msg: String| -> Result<minijinja::Value, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_function(
            "strftime_now",
            |_fmt: String| -> Result<String, minijinja::Error> { Ok(String::new()) },
        );

        let has_template = match model.get_chat_template(16 * 1024) {
            Ok(s) if !s.is_empty() => match env.add_template_owned("chat", s) {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "ChatTemplate: failed to compile model's Jinja chat template; \
                         falling back to llama_chat_apply_template (no kwargs support)"
                    );
                    false
                }
            },
            Ok(_) => {
                tracing::warn!(
                    "ChatTemplate: model has no embedded chat_template; \
                     falling back to llama_chat_apply_template"
                );
                false
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "ChatTemplate: get_chat_template failed; \
                     falling back to llama_chat_apply_template"
                );
                false
            }
        };

        Self { env, has_template }
    }

    fn render(
        &self,
        messages: &[ChatMsg],
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: Option<&serde_json::Value>,
        tool_choice: Option<&serde_json::Value>,
    ) -> Option<String> {
        if !self.has_template {
            return None;
        }

        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let tmpl = match self.env.get_template("chat") {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "ChatTemplate: get_template failed");
                return None;
            }
        };

        let tools_val = tools.cloned().unwrap_or(serde_json::Value::Null);
        let tool_choice_val = tool_choice.cloned().unwrap_or(serde_json::Value::Null);

        let ctx = minijinja::context! {
            messages => msgs,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
            tools => tools_val,
            tool_choice => tool_choice_val,
            bos_token => "",
            eos_token => "",
        };

        match tmpl.render(ctx) {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "ChatTemplate: Jinja render failed; \
                     falling back to llama_chat_apply_template for this turn"
                );
                None
            }
        }
    }
}

/// Streaming filter that strips `<think>...</think>` blocks.
struct ThinkStripper {
    in_think: bool,
    buffer: String,
}

impl ThinkStripper {
    const OPEN: &'static str = "<think>";
    const CLOSE: &'static str = "</think>";

    fn new() -> Self {
        Self {
            in_think: false,
            buffer: String::new(),
        }
    }

    fn push(&mut self, piece: &str) -> Option<String> {
        if piece.is_empty() {
            return None;
        }
        self.buffer.push_str(piece);

        let mut out = String::new();
        loop {
            if self.in_think {
                if let Some(idx) = self.buffer.find(Self::CLOSE) {
                    self.buffer.drain(..idx + Self::CLOSE.len());
                    self.in_think = false;
                    continue;
                }
                let keep = Self::CLOSE.len().saturating_sub(1);
                if self.buffer.len() > keep {
                    let drop_to = self.buffer.len() - keep;
                    let drop_to = (0..=drop_to)
                        .rev()
                        .find(|&i| self.buffer.is_char_boundary(i))
                        .unwrap_or(0);
                    self.buffer.drain(..drop_to);
                }
                break;
            }

            if let Some(idx) = self.buffer.find(Self::OPEN) {
                out.push_str(&self.buffer[..idx]);
                self.buffer.drain(..idx + Self::OPEN.len());
                self.in_think = true;
                continue;
            }

            let safe_end = self.buffer.len().saturating_sub(Self::OPEN.len() - 1);
            let safe_end = (0..=safe_end)
                .rev()
                .find(|&i| self.buffer.is_char_boundary(i))
                .unwrap_or(0);
            if safe_end > 0 {
                let head: String = self.buffer.drain(..safe_end).collect();
                out.push_str(&head);
            }
            break;
        }

        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    fn flush(&mut self) -> Option<String> {
        if self.in_think {
            self.buffer.clear();
            return None;
        }
        if self.buffer.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.buffer))
    }
}

/// One tool call extracted from a streamed `<tool_call>...</tool_call>` block.
#[derive(Debug, Clone)]
struct ParsedToolCall {
    name: String,
    arguments_json: String,
}

/// Streaming filter that strips Qwen3 / Hermes / DeepSeek-style
/// `<tool_call>{...}</tool_call>` blocks.
struct ToolCallStripper {
    in_call: bool,
    buffer: String,
    calls: Vec<ParsedToolCall>,
}

impl ToolCallStripper {
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";

    fn new() -> Self {
        Self {
            in_call: false,
            buffer: String::new(),
            calls: Vec::new(),
        }
    }

    fn push(&mut self, piece: &str) -> Option<String> {
        if piece.is_empty() {
            return None;
        }
        self.buffer.push_str(piece);

        let mut out = String::new();
        loop {
            if self.in_call {
                if let Some(idx) = self.buffer.find(Self::CLOSE) {
                    let inner = self.buffer[..idx].trim().to_string();
                    self.buffer.drain(..idx + Self::CLOSE.len());
                    self.in_call = false;
                    // If `parse_inner` can't parse the body as a JSON
                    // tool call, it returns the raw text so we can
                    // re-emit it on the regular text channel. Without
                    // this, models that emit malformed tool-call
                    // wrappers (Llama-3-style `<function=…>` inside a
                    // Qwen `<tool_call>` envelope, etc.) get their
                    // entire reply silently dropped.
                    if let Some(raw) = self.parse_inner(&inner) {
                        out.push_str(&raw);
                    }
                    continue;
                }
                break;
            }

            if let Some(idx) = self.buffer.find(Self::OPEN) {
                out.push_str(&self.buffer[..idx]);
                self.buffer.drain(..idx + Self::OPEN.len());
                self.in_call = true;
                continue;
            }

            let safe_end = self.buffer.len().saturating_sub(Self::OPEN.len() - 1);
            let safe_end = (0..=safe_end)
                .rev()
                .find(|&i| self.buffer.is_char_boundary(i))
                .unwrap_or(0);
            if safe_end > 0 {
                let head: String = self.buffer.drain(..safe_end).collect();
                out.push_str(&head);
            }
            break;
        }

        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    fn flush(&mut self) -> Option<String> {
        if self.in_call {
            self.buffer.clear();
            return None;
        }
        if self.buffer.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.buffer))
    }

    /// Returns `Some(raw_text)` when the body can't be turned into a
    /// dispatchable tool call. The caller re-emits that text on the
    /// regular text channel — losing the call (no tool can dispatch
    /// it) but preserving whatever the model intended to say.
    ///
    /// Returns `None` when the body parsed cleanly and was appended to
    /// `self.calls`.
    fn parse_inner(&mut self, inner: &str) -> Option<String> {
        let v: serde_json::Value = match serde_json::from_str(inner) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    raw = %inner,
                    "[llm] llama.cpp <tool_call> body did not parse as JSON; \
                     re-emitting as plain text"
                );
                return Some(inner.to_string());
            }
        };
        let name = v
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();
        if name.is_empty() {
            tracing::warn!(
                raw = %inner,
                "[llm] llama.cpp <tool_call> body had no `name` field; \
                 re-emitting as plain text"
            );
            return Some(inner.to_string());
        }
        let arguments_json = match v.get("arguments") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => "{}".to_string(),
        };
        self.calls.push(ParsedToolCall {
            name,
            arguments_json,
        });
        None
    }

    fn take_calls(&mut self) -> Vec<ParsedToolCall> {
        std::mem::take(&mut self.calls)
    }
}

/// Streaming consumer: drives a live `mpsc::Receiver<TurnEvent>`
/// through a [`ToolCallStripper`], emits surviving text + activation
/// taps to `callback` as they arrive, and dispatches each parsed
/// tool call via [`crate::tool_dispatch::dispatch_tool_call`].
///
/// Plugin-side simplification: the in-tree version takes an optional
/// `Arc<CancelGate>` and speculatively/permanently protects across
/// `<tool_call>` open / non-cancelable dispatch. That gate isn't
/// exposed to plugins, so this version just dispatches synchronously
/// on the streaming callback and relies on the universal future-drop
/// cancellation pathway to bound the in-flight window.
async fn stream_events_with_tool_dispatch<F>(
    mut events: mpsc::Receiver<TurnEvent>,
    tool_registry: &[ToolSpec],
    output_channel: &str,
    callback: &mut F,
) -> Result<usize, Error>
where
    F: FnMut(RuntimeData) -> Result<(), Error> + Send,
{
    use crate::tool_dispatch::{dispatch_tool_call, ToolCallAccum};

    let mut count = 0usize;
    let mut call_idx = 0usize;
    let mut stripper = ToolCallStripper::new();
    let mut empty_registry_warned = false;

    while let Some(event) = events.recv().await {
        match event {
            TurnEvent::Chunk(piece) => {
                let emitted = stripper.push(&piece);

                if let Some(text) = emitted {
                    callback(RuntimeData::Text(text))?;
                    count += 1;
                }

                let new_calls = stripper.take_calls();
                if !new_calls.is_empty() && tool_registry.is_empty() && !empty_registry_warned {
                    tracing::warn!(
                        n_calls = new_calls.len(),
                        "[llm] llama.cpp model emitted <tool_call> blocks but the \
                         node has no tools registered; dispatcher will drop them all"
                    );
                    empty_registry_warned = true;
                }
                for call in new_calls {
                    let accum = ToolCallAccum {
                        id: format!("call_{}", call_idx),
                        name: call.name,
                        arguments: call.arguments_json,
                    };
                    call_idx += 1;
                    let mut wrapped = |rd: RuntimeData| -> Result<(), Error> {
                        count += 1;
                        callback(rd)
                    };
                    dispatch_tool_call(tool_registry, &accum, output_channel, &mut wrapped)?;
                }
            }
            TurnEvent::Tap(t) => {
                callback(tap_to_runtime_data(t))?;
                count += 1;
            }
            TurnEvent::Error(e) => {
                return Err(e);
            }
        }
    }

    if let Some(text) = stripper.flush() {
        callback(RuntimeData::Text(text))?;
        count += 1;
    }
    for call in stripper.take_calls() {
        let accum = ToolCallAccum {
            id: format!("call_{}", call_idx),
            name: call.name,
            arguments: call.arguments_json,
        };
        call_idx += 1;
        let mut wrapped = |rd: RuntimeData| -> Result<(), Error> {
            count += 1;
            callback(rd)
        };
        dispatch_tool_call(tool_registry, &accum, output_channel, &mut wrapped)?;
    }

    Ok(count)
}

struct TurnStats {
    n_decoded: usize,
    n_reused: usize,
    n_response_tokens: u32,
}

/// Run one chat turn. Always clears the KV cache and re-decodes the full
/// formatted conversation, then samples the assistant response.
fn run_turn_incremental(
    state: &mut ChatState,
    template: &ChatTemplate,
    model: &llama_cpp_4::model::LlamaModel,
    ctx: &mut llama_cpp_4::context::LlamaContext,
    config: &LlamaCppGenerationConfig,
    user_text: &str,
    tools_json: Option<&serde_json::Value>,
    tool_choice: Option<&serde_json::Value>,
    mut tap_capture: Option<&mut llama_cpp_4::context::tensor_capture::TensorCapture>,
    event_tx: &mpsc::Sender<TurnEvent>,
    cancel_flag: Arc<AtomicBool>,
) -> Result<TurnStats, Error> {
    use llama_cpp_4::llama_batch::LlamaBatch;
    use llama_cpp_4::model::{AddBos, Special};
    use llama_cpp_4::sampling::LlamaSampler;
    use std::time::Instant;

    let turn_started = Instant::now();
    let tap_layer = config.activation_tap.as_ref().map(|t| t.layer);
    let tap_every_n = config
        .activation_tap
        .as_ref()
        .map(|t| t.every_n_tokens)
        .unwrap_or(0);

    let send_chunk = |s: String| -> Result<(), Error> {
        event_tx
            .blocking_send(TurnEvent::Chunk(s))
            .map_err(|_| Error::Execution("LlamaCpp client dropped event channel".into()))
    };
    let send_tap = |t: TappedActivation| -> Result<(), Error> {
        event_tx
            .blocking_send(TurnEvent::Tap(t))
            .map_err(|_| Error::Execution("LlamaCpp client dropped event channel".into()))
    };

    let mut probe_messages: Vec<ChatMsg> = state.messages.clone();
    probe_messages.push(ChatMsg {
        role: "user".to_string(),
        content: user_text.to_string(),
    });

    let formatted = match template.render(
        &probe_messages,
        true,
        false,
        tools_json,
        tool_choice,
    ) {
        Some(s) => s,
        None => {
            if tools_json.is_some() {
                tracing::warn!(
                    "ChatTemplate render failed with tools set; \
                     falling back to llama_chat_apply_template (drops tools). \
                     Model will not see the tools array this turn."
                );
            }
            let llama_msgs = to_llama_messages(&probe_messages)?;
            model
                .apply_chat_template(None, &llama_msgs, true)
                .map_err(|e| Error::Execution(format!("chat template apply: {}", e)))?
        }
    };

    let add_bos = if model.add_bos_token() {
        AddBos::Always
    } else {
        AddBos::Never
    };
    let prompt_tokens = model
        .str_to_token(&formatted, add_bos)
        .map_err(|e| Error::Execution(format!("tokenize: {}", e)))?;
    let n_prompt = prompt_tokens.len();
    if n_prompt == 0 {
        return Ok(TurnStats {
            n_decoded: 0,
            n_reused: 0,
            n_response_tokens: 0,
        });
    }

    ctx.clear_kv_cache();

    let mut batch = LlamaBatch::new(config.batch_size as usize, 1);
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let last = i == n_prompt - 1;
        batch
            .add(tok, i as i32, &[0], last)
            .map_err(|e| Error::Execution(format!("batch add (prefill): {}", e)))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| Error::Execution(format!("decode (prefill): {}", e)))?;

    if let (Some(layer), Some(cap)) = (tap_layer, tap_capture.as_deref_mut()) {
        if let Some(snap) = snapshot_last_token_hidden(cap, layer, "input", 0, turn_started) {
            send_tap(snap)?;
        }
    }

    let mut pos = n_prompt as i32;

    let mut chain_top_k: Vec<LlamaSampler> = Vec::new();
    let mut chain_no_top_k: Vec<LlamaSampler> = Vec::new();
    if config.repeat_penalty != 1.0 {
        chain_top_k.push(LlamaSampler::penalties_simple(64, config.repeat_penalty));
        chain_no_top_k.push(LlamaSampler::penalties_simple(64, config.repeat_penalty));
    }
    if config.min_p > 0.0 {
        chain_top_k.push(LlamaSampler::min_p(config.min_p, 1));
        chain_no_top_k.push(LlamaSampler::min_p(config.min_p, 1));
    }
    chain_top_k.extend([
        LlamaSampler::top_k(config.top_k as i32),
        LlamaSampler::top_p(config.top_p, 1),
        LlamaSampler::temp(config.temperature),
        LlamaSampler::dist(config.seed as u32),
    ]);
    chain_no_top_k.extend([
        LlamaSampler::top_p(config.top_p, 1),
        LlamaSampler::temp(config.temperature),
        LlamaSampler::dist(config.seed as u32),
    ]);
    let sampler = if config.top_k > 0 {
        LlamaSampler::chain_simple(chain_top_k)
    } else {
        LlamaSampler::chain_simple(chain_no_top_k)
    };

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut stripper = ThinkStripper::new();
    let mut response = String::new();

    let mut response_token_idx: u32 = 0;
    for _ in 0..config.max_tokens {
        if cancel_flag.load(Ordering::Relaxed) {
            tracing::info!("Generation turn cancelled mid-flight!");
            break;
        }
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        if model.is_eog_token(token) {
            break;
        }
        let bytes = model
            .token_to_bytes(token, Special::Tokenize)
            .map_err(|e| Error::Execution(format!("token decode: {}", e)))?;
        let cap = decoder
            .max_utf8_buffer_length(bytes.len())
            .unwrap_or(bytes.len() * 4 + 4);
        let mut piece = String::with_capacity(cap);
        let _ = decoder.decode_to_string(&bytes, &mut piece, false);

        if let Some(out) = stripper.push(&piece) {
            response.push_str(&out);
            send_chunk(out)?;
        }

        batch.clear();
        batch
            .add(token, pos, &[0], true)
            .map_err(|e| Error::Execution(format!("batch add (gen): {}", e)))?;
        ctx.decode(&mut batch)
            .map_err(|e| Error::Execution(format!("decode (gen): {}", e)))?;
        pos += 1;
        response_token_idx = response_token_idx.saturating_add(1);

        if let (Some(layer), Some(cap)) = (tap_layer, tap_capture.as_deref_mut()) {
            if tap_every_n > 0 && response_token_idx % tap_every_n == 0 {
                if let Some(snap) = snapshot_last_token_hidden(
                    cap,
                    layer,
                    "response",
                    response_token_idx,
                    turn_started,
                ) {
                    send_tap(snap)?;
                }
            }
        }
    }

    let cap = decoder.max_utf8_buffer_length(0).unwrap_or(8);
    let mut tail = String::with_capacity(cap);
    let _ = decoder.decode_to_string(&[], &mut tail, true);
    if let Some(out) = stripper.push(&tail) {
        response.push_str(&out);
        send_chunk(out)?;
    }
    if let Some(out) = stripper.flush() {
        response.push_str(&out);
        send_chunk(out)?;
    }

    // End-of-response sentinel.
    send_chunk("<|text_end|>".to_string())?;

    let user_msg = probe_messages
        .pop()
        .expect("probe_messages always has at least the new user msg");
    if !response.is_empty() {
        state.messages.push(user_msg);
        state.messages.push(ChatMsg {
            role: "assistant".to_string(),
            content: response,
        });
    }

    Ok(TurnStats {
        n_decoded: n_prompt,
        n_reused: 0,
        n_response_tokens: response_token_idx,
    })
}

fn snapshot_last_token_hidden(
    capture: &mut llama_cpp_4::context::tensor_capture::TensorCapture,
    layer: u32,
    phase: &'static str,
    token_index: u32,
    turn_started: std::time::Instant,
) -> Option<TappedActivation> {
    let tensor_name = format!("l_out-{}", layer);
    let info = capture.get(&tensor_name)?;
    let n_embd = info.n_embd() as usize;
    let n_tok = info.n_tokens() as usize;
    if n_embd == 0 || n_tok == 0 {
        return None;
    }
    let last = n_tok - 1;
    let start = last * n_embd;
    let end = start + n_embd;
    if end > info.data.len() {
        return None;
    }
    let hidden: Vec<f32> = info.data[start..end].to_vec();
    Some(TappedActivation {
        layer,
        hidden,
        phase,
        token_index,
        turn_offset_ms: turn_started.elapsed().as_millis() as u64,
    })
}

#[async_trait::async_trait]
impl InterruptableBackend for LlamaCppGenerationNode {
    async fn request_cancel(&self, session_id: &str) -> Result<(), Error> {
        let mut interrupts = self.interrupts.lock().unwrap();
        let flag = interrupts
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(AtomicBool::new(false)));
        flag.store(true, Ordering::Relaxed);
        Ok(())
    }
}

#[async_trait::async_trait]
impl StatefulConversationBackend for LlamaCppGenerationNode {
    async fn reset_history(&self, session_id: &str) -> Result<(), Error> {
        let sessions = self.sessions.lock().unwrap();
        if let Some(chat) = sessions.get(session_id) {
            let mut guard = chat.lock().unwrap();
            guard.messages.clear();
        }
        Ok(())
    }

    async fn set_context(&self, session_id: &str, context: &str) -> Result<(), Error> {
        let mut sessions = self.sessions.lock().unwrap();
        let chat = sessions
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(ChatState::new())));
        let mut guard = chat.lock().unwrap();
        if guard.messages.is_empty() {
            guard.messages.push(ChatMsg {
                role: "system".into(),
                content: context.into(),
            });
        } else if guard.messages[0].role == "system" {
            guard.messages[0].content = format!(
                "{}\nContext: {}",
                self.config.system_prompt.as_deref().unwrap_or(""),
                context
            );
        }
        Ok(())
    }

    async fn finalize_turn(
        &self,
        session_id: &str,
        interrupted: bool,
        elapsed_secs: f32,
    ) -> Result<(), Error> {
        if interrupted && elapsed_secs < 3.0 {
            let sessions = self.sessions.lock().unwrap();
            if let Some(chat) = sessions.get(session_id) {
                let mut guard = chat.lock().unwrap();
                if guard.messages.len() >= 2 {
                    guard.messages.pop();
                    if guard.messages.last().map(|m| m.role == "user").unwrap_or(false) {
                        guard.messages.pop();
                    }
                }
            }
        }
        Ok(())
    }
}