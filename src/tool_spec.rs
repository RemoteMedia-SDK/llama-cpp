//! LLM tool-call schema + side-effect dispatch hints.
//!
//! **Inlined** from `remotemedia-core::nodes::tool_spec` — the host
//! module isn't reachable from a Path-3 plugin (no link against
//! `remotemedia-core`), but the generation node's chat-template
//! integration and tool-dispatcher need the same `ToolSpec` shape so
//! prompts behave identically across the in-process llama.cpp path and
//! any other LLM transport that uses the same registry.
//!
//! Schema dropped: the host source uses `schemars::JsonSchema` for
//! manifest validation; we don't render schemas at the FFI boundary, so
//! the derive is omitted here (avoids pulling `schemars` into the
//! plugin's dep tree).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Dispatch contract for a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// Tool is consumed inline by the LLM node — the call IS the
    /// output. No result is fed back to the model.
    SideEffect,
    /// Tool's return value should be fed back to the model on a second
    /// generation pass. Not implemented in the Rust SSE pipeline yet.
    ReturnValue,
}

impl Default for ToolKind {
    fn default() -> Self {
        Self::SideEffect
    }
}

/// Schema + dispatch hint for a tool the LLM may call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(default)]
    pub kind: ToolKind,

    /// Whether barge-in is allowed to cancel mid-tool-call. See the
    /// host-side docs in `crates/core/src/nodes/tool_spec.rs` for the
    /// full lifecycle. The plugin honours this field via the
    /// `ToolCallStripper` → `dispatch_tool_call` chain — see
    /// `src/llama_cpp/generation.rs::stream_events_with_tool_dispatch`.
    /// **Note**: the host's `CancelGate` machinery is not reachable
    /// from this plugin, so non-cancelable tools currently rely on
    /// the universal future-drop suppression. Behavioural difference
    /// from the in-tree path is minimal because dispatch is
    /// synchronous on the streaming callback.
    #[serde(default = "default_true")]
    pub cancelable: bool,
}

fn default_true() -> bool {
    true
}

impl ToolSpec {
    /// Render as one entry in an OpenAI chat-completions `tools`
    /// array: `{ "type": "function", "function": { name, description,
    /// parameters } }`.
    pub fn to_openai_function(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// Render a slice of specs as a JSON array suitable for the
/// chat-completions `tools` request field.
pub fn to_openai_tools_array(specs: &[ToolSpec]) -> Value {
    Value::Array(specs.iter().map(ToolSpec::to_openai_function).collect())
}

/// Built-in `say` tool. Description identical to the host's.
pub fn default_say_tool() -> ToolSpec {
    ToolSpec {
        name: "say".to_string(),
        description: "Speak a sentence aloud to the user. The REQUIRED `text` \
parameter is the exact words to speak — if you omit it or \
leave it empty, nothing is synthesised and the user hears \
silence. Put the actual words inside the tool call; never \
write them after it.\n\n\
Correct: say(text=\"Hi Mathieu, here's your script.\")\n\
Wrong:   say()  followed by text outside the call.\n\n\
Use `say` for anything the user should HEAR: greetings, \
conversational answers, short summaries, confirmations. \
Use plain prose only — no markdown, no code, no lists."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description":
                        "The words to speak aloud. MUST be a non-empty \
        string of plain prose. Example: \"Sure thing, here's the Python script.\"",
                    "minLength": 1
                }
            },
            "required": ["text"]
        }),
        kind: ToolKind::SideEffect,
        cancelable: true,
    }
}

/// Built-in `show` tool.
pub fn default_show_tool() -> ToolSpec {
    ToolSpec {
        name: "show".to_string(),
        description: "Display written content to the user as markdown. The REQUIRED \
`content` parameter is the markdown text itself — if you omit \
it or leave it empty, nothing is rendered. Put all written \
content inside the tool call; never write it after the call.\n\n\
Correct: show(content=\"```python\\ndef hi(): ...\\n```\")\n\
Wrong:   show()  followed by markdown outside the call.\n\n\
Use `show` for anything the user should READ rather than hear: \
code blocks (triple-backtick fences with a language tag), \
tables, lists, file paths, long explanations, command output."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description":
                        "The markdown text to render. MUST be a non-empty \
        string. Example: \"```python\\nprint('hi')\\n```\"",
                    "minLength": 1
                }
            },
            "required": ["content"]
        }),
        kind: ToolKind::SideEffect,
        cancelable: true,
    }
}
