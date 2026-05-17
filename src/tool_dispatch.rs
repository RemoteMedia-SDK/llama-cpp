//! Streaming tool-call accumulation and side-effect dispatch.
//!
//! **Inlined** from `remotemedia-core::llm::tool_dispatch` — the host
//! module isn't reachable from a Path-3 plugin, but the generation
//! node's streaming dispatch path needs `dispatch_tool_call` /
//! `ToolCallAccum` to consume `<tool_call>{...}</tool_call>` blocks
//! emitted by Qwen3 / Hermes / DeepSeek GGUFs.

use remotemedia_plugin_sdk::types::{tag_text_str, Error, RuntimeData, TEXT_CHANNEL_DEFAULT};
use serde_json::Value;

use crate::tool_spec::{ToolKind, ToolSpec};

/// Per-index accumulator for `delta.tool_calls` deltas.
#[derive(Debug, Default)]
pub struct ToolCallAccum {
    /// Streaming `tool_call.id`.
    pub id: String,
    pub name: String,
    /// Stringified JSON fragment that, once concatenated, parses to
    /// the tool-call argument object.
    pub arguments: String,
}

fn lookup_tool<'a>(registry: &'a [ToolSpec], name: &str) -> Option<&'a ToolSpec> {
    registry.iter().find(|t| t.name == name)
}

/// Dispatch one accumulated tool call. See the host docs for the full
/// routing matrix; this is a verbatim port.
pub fn dispatch_tool_call<F>(
    registry: &[ToolSpec],
    call: &ToolCallAccum,
    output_channel: &str,
    callback: &mut F,
) -> Result<(), Error>
where
    F: FnMut(RuntimeData) -> Result<(), Error>,
{
    if call.name.is_empty() {
        tracing::warn!("[llm] tool call with no name received; dropping");
        return Ok(());
    }

    let spec = match lookup_tool(registry, &call.name) {
        Some(s) => s,
        None => {
            tracing::warn!(
                tool = %call.name,
                "[llm] model called unregistered tool; dropping"
            );
            return Ok(());
        }
    };

    if spec.kind == ToolKind::ReturnValue {
        tracing::warn!(
            tool = %call.name,
            "[llm] return_value tools require a second generation pass \
             (not yet implemented in the streaming path); skipping"
        );
        return Ok(());
    }

    let args: Value = serde_json::from_str(&call.arguments).unwrap_or_else(|e| {
        tracing::warn!(
            tool = %call.name,
            error = %e,
            raw = %call.arguments,
            "[llm] tool call arguments did not parse as JSON; treating as empty"
        );
        Value::Object(serde_json::Map::new())
    });

    let extract_string = |keys: &[&str]| -> Option<String> {
        for k in keys {
            if let Some(s) = args.get(*k).and_then(Value::as_str) {
                if !s.is_empty() {
                    return Some(s.to_string());
                }
            }
        }
        if let Value::String(s) = &args {
            if !s.is_empty() {
                return Some(s.clone());
            }
        }
        None
    };

    match call.name.as_str() {
        "say" => {
            let spoken = extract_string(&["text", "content", "message", "body", "spoken"]);
            if let Some(text) = spoken {
                let flushable = if text.ends_with('\n') {
                    text
                } else {
                    format!("{}\n", text)
                };
                callback(RuntimeData::Text(tag_text_str(&flushable, output_channel)))?;
                if output_channel != TEXT_CHANNEL_DEFAULT {
                    callback(RuntimeData::Text(tag_text_str(
                        &flushable,
                        TEXT_CHANNEL_DEFAULT,
                    )))?;
                }
            } else {
                tracing::warn!(
                    args = %call.arguments,
                    "[llm] `say` tool call had no recognisable text arg; nothing to synthesise"
                );
            }
        }
        "show" => {
            let written = extract_string(&["content", "markdown", "text", "body"]);
            if let Some(text) = written {
                callback(RuntimeData::Text(tag_text_str(&text, "ui")))?;
            } else {
                tracing::warn!(
                    args = %call.arguments,
                    "[llm] `show` tool call had no recognisable content arg"
                );
            }
        }
        "perform_motion" => {
            let prompt = extract_string(&["prompt", "description", "action"]);
            if let Some(p) = prompt {
                let env = serde_json::json!({
                    "kind": "motion_intent",
                    "prompt": p,
                });
                callback(RuntimeData::Json(env))?;
            } else {
                tracing::warn!(
                    args = %call.arguments,
                    "[llm] `perform_motion` tool call had no recognisable prompt arg"
                );
            }
        }
        other => {
            tracing::debug!(
                tool = %other,
                args = %call.arguments,
                "[llm] side_effect tool dispatched; no built-in handler — dropping"
            );
        }
    }
    Ok(())
}
