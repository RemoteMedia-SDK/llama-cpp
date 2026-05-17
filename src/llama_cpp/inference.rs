//! Blocking inference helpers for llama.cpp.
//!
//! llama.cpp types (LlamaBatch, LlamaContext, LlamaModel) contain raw C pointers
//! and are not `Send`. This module provides blocking functions that run on a
//! dedicated thread via `tokio::task::spawn_blocking`.

use std::num::NonZeroU32;

use remotemedia_plugin_sdk::types::Error;

use super::config::{ActivationPooling, GpuOffload, LlamaCppGenerationConfig};

/// Result of a generation task.
#[derive(Debug)]
pub struct GenerationResult {
    /// Generated text chunks.
    pub chunks: Vec<String>,
    /// Number of tokens generated.
    pub n_tokens: usize,
}

/// Run text generation on a blocking thread.
pub fn run_generation(
    config: &LlamaCppGenerationConfig,
    prompt: &str,
) -> Result<GenerationResult, Error> {
    use llama_cpp_4::model::{params::LlamaModelParams, LlamaModel};

    let backend = super::backend::get_or_init()?;

    let n_gpu_layers = match config.backend.gpu_offload {
        GpuOffload::None => 0,
        GpuOffload::All => 1000,
        GpuOffload::Layers(n) => n as u32,
    };
    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

    let model = LlamaModel::load_from_file(backend, &config.model_path, &model_params)
        .map_err(|e| Error::Execution(format!("Model load failed: {}", e)))?;

    let mut ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    ctx_params = ctx_params.with_n_ctx(NonZeroU32::new(config.context_size));
    ctx_params = ctx_params.with_n_batch(config.batch_size);
    if config.backend.flash_attention {
        ctx_params = ctx_params.with_flash_attention(true);
    }
    if let Some(threads) = config.backend.threads {
        ctx_params = ctx_params.with_n_threads(threads as i32);
    }

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| Error::Execution(format!("Context creation failed: {}", e)))?;

    run_generation_with_ctx_inner(&model, &mut ctx, config, prompt)
}

/// Run text generation against a pre-loaded model + context.
pub fn run_generation_with_ctx(
    model: &llama_cpp_4::model::LlamaModel,
    ctx: &mut llama_cpp_4::context::LlamaContext,
    config: &LlamaCppGenerationConfig,
    prompt: &str,
) -> Result<Vec<String>, Error> {
    ctx.clear_kv_cache();
    run_generation_with_ctx_inner(model, ctx, config, prompt).map(|r| r.chunks)
}

fn run_generation_with_ctx_inner(
    model: &llama_cpp_4::model::LlamaModel,
    ctx: &mut llama_cpp_4::context::LlamaContext,
    config: &LlamaCppGenerationConfig,
    prompt: &str,
) -> Result<GenerationResult, Error> {
    use llama_cpp_4::llama_batch::LlamaBatch;
    use llama_cpp_4::model::{AddBos, LlamaChatMessage, Special};
    use llama_cpp_4::sampling::LlamaSampler;

    let mut messages: Vec<LlamaChatMessage> = Vec::new();
    if let Some(sys) = config.system_prompt.as_ref().filter(|s| !s.is_empty()) {
        messages.push(
            LlamaChatMessage::new("system".to_string(), sys.clone())
                .map_err(|e| Error::Execution(format!("Invalid system prompt: {}", e)))?,
        );
    }
    messages.push(
        LlamaChatMessage::new("user".to_string(), prompt.to_string())
            .map_err(|e| Error::Execution(format!("Invalid user prompt: {}", e)))?,
    );

    let formatted = model
        .apply_chat_template(None, &messages, true)
        .map_err(|e| Error::Execution(format!("Chat template apply failed: {}", e)))?;

    let tokens = model
        .str_to_token(&formatted, AddBos::Always)
        .map_err(|e| Error::Execution(format!("Tokenization failed: {}", e)))?;

    let n_prompt = tokens.len();
    if n_prompt == 0 {
        return Ok(GenerationResult {
            chunks: Vec::new(),
            n_tokens: 0,
        });
    }

    let mut batch = LlamaBatch::new(config.batch_size as usize, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        let last = i == n_prompt - 1;
        batch
            .add(tok, i as i32, &[0], last)
            .map_err(|e| Error::Execution(format!("Batch add failed: {}", e)))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| Error::Execution(format!("Prefill decode failed: {}", e)))?;

    let mut chain_top_k: Vec<LlamaSampler> = Vec::new();
    let mut chain_no_top_k: Vec<LlamaSampler> = Vec::new();

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

    let mut pos = n_prompt as i32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut chunks = Vec::new();
    let mut n_tokens = 0usize;

    for _ in 0..config.max_tokens {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);

        if model.is_eog_token(token) {
            break;
        }

        let bytes = model
            .token_to_bytes(token, Special::Plaintext)
            .map_err(|e| Error::Execution(format!("Token decode failed: {}", e)))?;

        let mut piece = String::new();
        decoder.decode_to_string(&bytes, &mut piece, false);

        if !piece.is_empty() {
            chunks.push(piece);
            n_tokens += 1;
        }

        batch.clear();
        batch
            .add(token, pos, &[0], true)
            .map_err(|e| Error::Execution(format!("Batch add failed: {}", e)))?;

        ctx.decode(&mut batch)
            .map_err(|e| Error::Execution(format!("Decode failed: {}", e)))?;

        pos += 1;
    }

    Ok(GenerationResult { chunks, n_tokens })
}

/// Result of an embedding task.
#[derive(Debug)]
pub struct EmbeddingResult {
    pub embedding: Vec<f32>,
    pub hidden_size: usize,
}

pub fn run_embedding(
    model_path: &str,
    text: &str,
    context_size: u32,
    batch_size: u32,
    gpu_offload: GpuOffload,
    flash_attention: bool,
    threads: Option<u32>,
) -> Result<EmbeddingResult, Error> {
    use llama_cpp_4::llama_batch::LlamaBatch;
    use llama_cpp_4::model::{params::LlamaModelParams, AddBos, LlamaModel};

    let backend = super::backend::get_or_init()?;

    let mut model_params = LlamaModelParams::default();
    match gpu_offload {
        GpuOffload::None => {}
        GpuOffload::All => {
            model_params = model_params.with_n_gpu_layers(99);
        }
        GpuOffload::Layers(n) => {
            model_params = model_params.with_n_gpu_layers(n as u32);
        }
    }

    let model = LlamaModel::load_from_file(backend, model_path, &model_params)
        .map_err(|e| Error::Execution(format!("Model load failed: {}", e)))?;

    let hidden_size = model.n_embd() as usize;

    let mut ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    ctx_params = ctx_params.with_n_ctx(NonZeroU32::new(context_size));
    ctx_params = ctx_params.with_n_batch(batch_size);
    ctx_params = ctx_params.with_embeddings(true);

    if flash_attention {
        ctx_params = ctx_params.with_flash_attention(true);
    }

    if let Some(threads) = threads {
        ctx_params = ctx_params.with_n_threads(threads as i32);
    }

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| Error::Execution(format!("Context creation failed: {}", e)))?;

    let tokens = model
        .str_to_token(text, AddBos::Always)
        .map_err(|e| Error::Execution(format!("Tokenization failed: {}", e)))?;

    let n_tokens = tokens.len();
    if n_tokens == 0 {
        return Ok(EmbeddingResult {
            embedding: vec![],
            hidden_size,
        });
    }

    let mut batch = LlamaBatch::new(batch_size as usize, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], true)
            .map_err(|e| Error::Execution(format!("Batch add failed: {}", e)))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| Error::Execution(format!("Decode failed: {}", e)))?;

    let embedding = ctx
        .embeddings_seq_ith(0)
        .map_err(|e| Error::Execution(format!("Embedding extraction failed: {}", e)))?;

    Ok(EmbeddingResult {
        embedding: embedding.to_vec(),
        hidden_size,
    })
}

/// Captured activation at a specific layer.
#[derive(Debug)]
pub struct ActivationCapture {
    pub layer: usize,
    pub activation: Vec<f32>,
    pub hidden_size: usize,
    pub raw_norm: f32,
}

pub fn run_activation(
    model_path: &str,
    text: &str,
    layers: &[usize],
    context_size: u32,
    batch_size: u32,
    gpu_offload: GpuOffload,
    flash_attention: bool,
    threads: Option<u32>,
    pooling: ActivationPooling,
) -> Result<Vec<ActivationCapture>, Error> {
    use llama_cpp_4::context::tensor_capture::TensorCapture;
    use llama_cpp_4::llama_batch::LlamaBatch;
    use llama_cpp_4::model::{params::LlamaModelParams, AddBos, LlamaModel};

    let backend = super::backend::get_or_init()?;

    let mut model_params = LlamaModelParams::default();
    match gpu_offload {
        GpuOffload::None => {}
        GpuOffload::All => {
            model_params = model_params.with_n_gpu_layers(99);
        }
        GpuOffload::Layers(n) => {
            model_params = model_params.with_n_gpu_layers(n as u32);
        }
    }

    let model = LlamaModel::load_from_file(backend, model_path, &model_params)
        .map_err(|e| Error::Execution(format!("Model load failed: {}", e)))?;

    let hidden_size = model.n_embd() as usize;

    let mut capture = TensorCapture::for_layers(layers);

    let mut ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    ctx_params = ctx_params.with_n_ctx(NonZeroU32::new(context_size));
    ctx_params = ctx_params.with_n_batch(batch_size);
    ctx_params = ctx_params.with_embeddings(true);
    ctx_params = ctx_params.with_tensor_capture(&mut capture);

    if flash_attention {
        ctx_params = ctx_params.with_flash_attention(true);
    }

    if let Some(threads) = threads {
        ctx_params = ctx_params.with_n_threads(threads as i32);
    }

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| Error::Execution(format!("Context creation failed: {}", e)))?;

    let tokens = model
        .str_to_token(text, AddBos::Always)
        .map_err(|e| Error::Execution(format!("Tokenization failed: {}", e)))?;

    let n_tokens = tokens.len();
    if n_tokens == 0 {
        return Ok(Vec::new());
    }

    let mut batch = LlamaBatch::new(batch_size as usize, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], true)
            .map_err(|e| Error::Execution(format!("Batch add failed: {}", e)))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| Error::Execution(format!("Decode failed: {}", e)))?;

    let mut results = Vec::new();

    for &layer in layers {
        let tensor_name = format!("l_out-{}", layer);
        if let Some(info) = capture.get(&tensor_name) {
            let data = &info.data;
            let n_embd = info.n_embd() as usize;
            let n_tok = info.n_tokens() as usize;

            let activations: Vec<Vec<f32>> = (0..n_tok)
                .filter_map(|t| {
                    let start = t * n_embd;
                    let end = start + n_embd;
                    if end <= data.len() {
                        Some(data[start..end].to_vec())
                    } else {
                        None
                    }
                })
                .collect();

            let pooled = pool_activations(&activations, pooling);
            let raw_norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();

            results.push(ActivationCapture {
                layer,
                activation: pooled,
                hidden_size,
                raw_norm,
            });
        }
    }

    Ok(results)
}

fn pool_activations(activations: &[Vec<f32>], pooling: ActivationPooling) -> Vec<f32> {
    if activations.is_empty() {
        return vec![];
    }

    let hidden_size = activations[0].len();

    match pooling {
        ActivationPooling::Mean => {
            let mut mean = vec![0.0f32; hidden_size];
            let count = activations.len() as f32;
            for act in activations {
                for (m, &v) in mean.iter_mut().zip(act) {
                    *m += v;
                }
            }
            for m in &mut mean {
                *m /= count;
            }
            mean
        }
        ActivationPooling::LastToken => activations.last().cloned().unwrap_or_default(),
        ActivationPooling::FirstToken => activations.first().cloned().unwrap_or_default(),
    }
}
