# llama-cpp — GGUF inference node family

Standalone Path 3 Rust cdylib that registers the four `LlamaCpp*` streaming
nodes into the
[RemoteMedia SDK](https://github.com/RemoteMedia-SDK/remotemedia-sdk)
streaming pipeline registry.

This plugin wraps the [llama.cpp](https://github.com/ggerganov/llama.cpp)
C library through the safe [`llama-cpp-4`](https://crates.io/crates/llama-cpp-4)
Rust bindings, providing in-process GGUF inference for text generation,
embeddings, hidden-state extraction, and steering.

## Build-time CUDA requirement

**This plugin links `llama-cpp-sys-4` with the `cuda` feature enabled.**
At build time you need the CUDA toolkit installed and visible to
`nvcc`. First build takes 15–30 minutes because the underlying
`llama-cpp-sys-4` C++/CUDA layer compiles ~180 template-instance
kernels (`fattn-mma-f16-*`, `mmq-instance-*`).

The plugin itself (`libllama_cpp_plugin.so`) is ~2.3 MiB because
`llama-cpp-sys-4` uses **dynamic linking** for the heavy GGML
backends. The companion shared libraries — built into the same
output directory — are:

| Library                  | Size      | Purpose                              |
|--------------------------|-----------|--------------------------------------|
| `libllama.so.0`          | ~1.5 MiB  | Top-level llama.cpp surface          |
| `libggml.so.0`           | ~60 KiB   | GGML loader                          |
| `libggml-base.so.0`      | ~870 KiB  | GGML base                            |
| `libggml-cpu.so.0`       | ~880 KiB  | CPU backend                          |
| **`libggml-cuda.so.0`**  | **~500 MiB** | **CUDA backend (heavy)**         |
| `libggml-blas.so.0`      | varies    | BLAS backend                         |
| `libllama-common.so.0`   | small     | Internal common helpers              |
| `libmtmd.so.0`           | small     | Multi-modal extensions               |

Consumers need all of these reachable via `$LD_LIBRARY_PATH` or in
the same directory as the plugin `.so`. The release pipeline
packages them together.

If you don't have a CUDA-capable build environment:

1. **Build without CUDA**: clone this repo, edit `Cargo.toml` to drop
   the `features = ["cuda"]` from both `llama-cpp-4` and
   `llama-cpp-sys-4`, then `cargo build --release`. The resulting
   plugin runs on CPU only; consumers pass `gpu_offload: "none"`.
2. **Use a prebuilt release asset**: the GitHub release pipeline
   (`.github/workflows/release.yml`) cross-builds platform-specific
   binaries that ship via `release-manifest.json`.

At **runtime** the loaded `.so` finds CUDA dynamically (`libcuda.so.1`,
`libcudart.so`) — consumers on a CUDA host don't need the toolkit, just
a current NVIDIA driver.

## Use from a manifest

```json
{
  "version": "v1",
  "plugins": ["llama-cpp@v0.2.0"],
  "nodes": [
    {
      "id": "llm",
      "node_type": "LlamaCppGenerationNode",
      "params": {
        "model_path": "/models/Qwen3.5-9B-MLX-4bit.gguf",
        "context_size": 4096,
        "max_tokens": 256,
        "temperature": 0.8,
        "system_prompt": "You are a helpful voice assistant.",
        "backend": { "gpu_offload": "all", "flash_attention": true }
      }
    }
  ]
}
```

The SDK resolver expands `llama-cpp@v0.2.0` to
`github.com/RemoteMedia-SDK/llama-cpp`, fetches `plugin.toml`, then
falls through to `release-manifest.json` for the platform-specific
prebuilt `.so` / `.dylib` / `.dll` asset.

## Build the cdylib locally

```bash
git clone https://github.com/RemoteMedia-SDK/llama-cpp
cd llama-cpp
cargo build --release
# → target/release/libllama_cpp_plugin.so  (~200 MiB)
```

First build takes 5–10 minutes because `llama-cpp-sys-4` compiles
several hundred MB of C/C++ + CUDA kernels.

## Releasing a new version

**GitHub Actions cannot cut releases for this plugin.** The standard
`build-plugin.yml` reusable workflow runs on `ubuntu-latest` /
`macos-latest` / `windows-latest`, none of which ship the CUDA
toolkit. Even with CUDA available, the dynamic-linked layout
(cdylib + ~8 companion `libggml-*.so.0` libraries) doesn't fit the
upstream workflow's single-binary upload contract. The tag-triggered
`.github/workflows/release.yml` is intentionally a tripwire that
fails with a pointer to this section.

Releases are cut **locally** on a CUDA-equipped host via the helper
script:

```bash
# 1. Bump the version
$EDITOR plugin.toml Cargo.toml   # version = "0.X.Y" in both
cargo update -p llama-cpp-plugin
git commit -am "release: v0.X.Y"

# 2. Tag and run the script
git tag v0.X.Y
scripts/release-local.sh v0.X.Y                  # builds + uploads
scripts/release-local.sh v0.X.Y --dry-run        # builds + stages locally, no upload
scripts/release-local.sh v0.X.Y --platform aarch64-linux   # other platforms
```

The script:

1. Verifies the tag matches `plugin.toml` / `Cargo.toml`.
2. Runs `cargo build --release`.
3. Locates the SONAME-versioned companion libraries
   (`libllama.so.0`, `libggml-{base,cpu,cuda,blas}.so.0`,
   `libllama-common.so.0`, `libmtmd.so.0`) in
   `target/llama-cmake-cache/<hash>/lib/`.
4. Stages them into a tarball named
   `lib{plugin}-{platform}-companions.tar.gz`.
5. Renames the cdylib to `lib{plugin}-{platform}.{ext}` per the
   resolver's `current_platform()` lookup.
6. SHA256-hashes both, writes `.sha256` sidecars.
7. Generates `release-manifest.json` with `file`, `sha256`,
   `companions`, `companions_sha256` keys per platform. Multi-platform
   releases accrete by re-running the script against the same tag on
   each host — the script downloads the existing manifest and merges
   the new platform entry.
8. Pushes the tag and runs `gh release create` (or
   `gh release upload --clobber` if the release already exists).

Consumers fetch the cdylib from the manifest's `file` entry and the
companion tarball from `companions`, extract the tarball next to the
cdylib (or anywhere on `LD_LIBRARY_PATH`), and dlopen the cdylib.

## What it exports

| Node type                  | Input                       | Output                                                       |
|----------------------------|-----------------------------|--------------------------------------------------------------|
| `LlamaCppGenerationNode`   | `Text` / `Json` prompt      | Streamed `Text` chunks + `Tensor` (activation taps) + tool envelopes |
| `LlamaCppEmbeddingNode`    | `Text` / `Json`             | `Tensor` (dense embedding vector)                            |
| `LlamaCppActivationNode`   | `Text` / `Json`             | `Tensor` (one per requested layer, pooled per `pooling`)     |
| `LlamaCppSteerNode`        | `Tensor` / `Text` / `Json`  | `Text` chunks + `Json` (steering delta metadata)             |

## Configuration highlights

### Generation

- **Activation tap**: capture hidden-state snapshots at any layer via
  `activation_tap: { layer: 15, every_n_tokens: 32 }`. Emits
  `Tensor` envelopes with `metadata.kind = "activation_tap"`
  alongside text chunks. Used by emotion-driven face nodes.
- **Tool calls**: built-in `say` / `show` tools plus arbitrary
  user-defined entries in `tools: [...]`. The streaming output is
  parsed for `<tool_call>{...}</tool_call>` blocks (Qwen3 / Hermes /
  DeepSeek format); each block is stripped from the text stream and
  dispatched.
- **Chat templates**: the GGUF's embedded Jinja chat template is
  rendered via [minijinja](https://crates.io/crates/minijinja) with
  `tools`, `tool_choice`, and `enable_thinking=false` kwargs so
  Qwen3-style templates produce the `# Tools` system block.

### Multi-Token Prediction (Qwen3.6 MTP)

Since `v0.2.0` the plugin links `llama-cpp-4` v0.3.0, which carries the
upstream MTP support merged in
[ggml-org/llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673).
Loading a Qwen3.6 MTP GGUF (e.g. `Qwen3.6-27B-Q4_K_M-mtp.gguf`,
`Qwen3.6-27B-IQ2_M-mtp.gguf`) works out of the box — the MTP head is
autodetected from GGUF metadata.

What you get today:

- **Drop-in load** of MTP-flavored Qwen3.6 GGUFs through
  `LlamaCppGenerationNode` / `LlamaCppEmbeddingNode` /
  `LlamaCppActivationNode`. Normal next-token generation runs against
  the base head; the MTP head is simply present in the file.

What is **not yet wired** through the plugin params:

- `LlamaContextType::Mtp` / `LlamaContextParams::with_ctx_type` for
  speculative-decoding draft contexts.
- The `llama_cpp_4::mtp::MtpSession` draft loop (would need a new
  `mtp: { enabled: true, n_max: 1, n_min: 0, p_min: 0.0 }` block on
  `LlamaCppGenerationNode` params and a second model handle).

Speculative-decoding throughput gains (≈+6–10% on Qwen3.6-27B per
upstream benchmarks) therefore require either (a) using the
`llama-cpp-4` API directly until the plugin exposes those params or
(b) opening an issue for the MTP draft-loop wiring.

### Embedding

- Pooling modes: `mean` (default), `last_token`, `first_token`, `cls`.
- L2 normalization on by default; turn off with `l2_normalize: false`.

### Activation extraction

- Captures any number of layers at once via `layers: [15, 21]`.
- Pooling modes: `last_token` (default), `mean`, `first_token`.
- Per-layer raw-norm metadata included on each `Tensor` envelope.

### Steering

- Accepts pre-extracted steering vectors as `RuntimeData::Tensor`
  envelopes labelled via `metadata.emotion` / `metadata.label`.
- Coefficients updated at runtime via `Json {"coefficients": {...}}`.
- **Metadata mode only**: the underlying KV-cache injection is not
  yet implemented (mirrors the in-tree implementation status). Text
  output passes through without hidden-state modification; steering
  delta + per-vector coefficients are surfaced in the output `Json`
  envelope.

## Plugin-side differences from the in-tree node

This is a mechanical port. Two structural changes worth calling out:

1. **No per-session typed `ChatState`**. The in-tree
   `LlamaCppGenerationNode` uses the host's
   `make_session_state` / `try_state` mechanism to keep multi-turn
   conversation history per session. That surface isn't reachable
   from the Path-3 FFI boundary, so the plugin constructs a fresh
   `ChatState` on every `generate()` call (still picks up
   `system_prompt`). Multi-turn coordination should happen outside
   the plugin — typically the consumer batches history into the
   prompt itself.
2. **No `CancelGate` integration**. The in-tree dispatcher takes a
   `ProtectGuard` whenever a `cancelable: false` tool dispatches so
   subsequent `barge_in` envelopes are suppressed for the rest of the
   turn. The plugin doesn't have access to `CancelGate` (it's a
   host-internal type), so the dispatcher just calls the tool
   dispatcher directly and relies on the universal future-drop
   cancellation pathway.

`<think>...</think>` stripping, `<tool_call>` parsing + dispatch,
the activation-tap side channel, the embedded-Jinja chat template
render path, and the `<|text_end|>` sentinel all behave identically.

## License

See `LICENSE.md`. Governed by the RemoteMedia SDK Community License 1.0.
