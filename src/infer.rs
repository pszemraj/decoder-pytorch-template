//! Standalone inference module for loading checkpoints and generating text.

use anyhow::{anyhow, Result};
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Int, TensorData},
};
use std::path::Path;

use crate::{
    config::ModelConfig,
    data::{ByteTokenizer, Tokenizer},
    models::llama::MpPolicy,
    models::LlamaModel,
    sampling::{sample_next_token, SamplingParams},
};

/// Load model configuration from a companion JSON file or fallback YAML.
///
/// Looks for `<checkpoint_path>.config.json` first, then falls back to
/// `config_override` if provided.
pub fn load_model_config(
    checkpoint_path: &Path,
    config_override: Option<&Path>,
) -> Result<ModelConfig> {
    // Try companion config file first
    let config_json_path = checkpoint_path.with_extension("config.json");
    if config_json_path.exists() {
        log::info!("Loading model config from: {}", config_json_path.display());
        let config_str = std::fs::read_to_string(&config_json_path)?;
        let config: ModelConfig = serde_json::from_str(&config_str)?;
        return Ok(config);
    }

    // Try checkpoint path with .bin replaced by .config.json
    let checkpoint_str = checkpoint_path.to_string_lossy();
    let alt_config_path = if checkpoint_str.ends_with(".bin") {
        let base = checkpoint_str.trim_end_matches(".bin");
        std::path::PathBuf::from(format!("{}.config.json", base))
    } else {
        checkpoint_path.with_extension("config.json")
    };

    if alt_config_path.exists() && alt_config_path != config_json_path {
        log::info!("Loading model config from: {}", alt_config_path.display());
        let config_str = std::fs::read_to_string(&alt_config_path)?;
        let config: ModelConfig = serde_json::from_str(&config_str)?;
        return Ok(config);
    }

    // Fall back to provided config override
    if let Some(yaml_path) = config_override {
        log::info!(
            "Loading model config from YAML override: {}",
            yaml_path.display()
        );
        let config_str = std::fs::read_to_string(yaml_path)?;

        // Try parsing as TrainingConfig first (common case)
        if let Ok(training_config) = serde_saphyr::from_str::<crate::TrainingConfig>(&config_str) {
            return Ok(training_config.model);
        }

        // Try parsing as bare ModelConfig
        let config: ModelConfig = serde_saphyr::from_str(&config_str)?;
        return Ok(config);
    }

    Err(anyhow!(
        "No model configuration found. Provide --config or ensure checkpoint has companion .config.json"
    ))
}

/// Load a model checkpoint from disk.
pub fn load_checkpoint<B: Backend>(
    checkpoint_path: &Path,
    model_config: &ModelConfig,
    device: &B::Device,
) -> Result<LlamaModel<B>> {
    log::info!("Loading checkpoint from: {}", checkpoint_path.display());

    // Create model with fp32 policy (inference doesn't need mixed precision)
    let model = LlamaModel::<B>::new(model_config.clone(), device, MpPolicy::fp32());

    // Load the record
    let checkpoint_str = checkpoint_path
        .to_str()
        .ok_or_else(|| anyhow!("Invalid checkpoint path"))?;

    // Strip .bin extension if present (Burn adds it automatically)
    let record_path = if checkpoint_str.ends_with(".bin") {
        checkpoint_str.trim_end_matches(".bin").to_string()
    } else {
        checkpoint_str.to_string()
    };

    let record = CompactRecorder::new()
        .load(record_path.into(), device)
        .map_err(|e| anyhow!("Failed to load checkpoint: {}", e))?;

    let model = model.load_record(record);
    log::info!("Checkpoint loaded successfully");

    Ok(model)
}

/// Generate text from a prompt.
pub fn generate<B: Backend>(
    model: &LlamaModel<B>,
    prompt: &str,
    max_length: usize,
    params: &SamplingParams,
    device: &B::Device,
) -> Result<String> {
    let tokenizer = ByteTokenizer;
    let prompt_tokens = tokenizer.encode(prompt);

    if prompt_tokens.is_empty() {
        return Ok(String::new());
    }

    let generated = generate_tokens(model, &prompt_tokens, max_length, params, device)?;

    // Decode only the newly generated tokens
    let completion_tokens = if generated.len() > prompt_tokens.len() {
        &generated[prompt_tokens.len()..]
    } else {
        &[]
    };

    Ok(tokenizer.decode(completion_tokens))
}

/// Generate tokens from a prompt.
pub fn generate_tokens<B: Backend>(
    model: &LlamaModel<B>,
    prompt: &[i64],
    max_length: usize,
    params: &SamplingParams,
    device: &B::Device,
) -> Result<Vec<i64>> {
    if prompt.is_empty() {
        return Ok(Vec::new());
    }

    let mut tokens = prompt.to_vec();
    let max_context = model.max_position_embeddings();

    for _ in 0..max_length {
        let start = tokens.len().saturating_sub(max_context);
        let context = &tokens[start..];
        if context.is_empty() {
            break;
        }

        let input = tokens_to_tensor::<B>(context, device).unsqueeze();
        let logits = model.forward(input, start);
        let [_, _, vocab_size] = logits.dims();
        let last_index = context.len() - 1;
        let last_logits = logits
            .slice([0..1, last_index..last_index + 1, 0..vocab_size])
            .squeeze_dims(&[0, 1]);

        let next_token = sample_next_token(last_logits, params, &tokens)?;
        tokens.push(next_token);
    }

    Ok(tokens)
}

fn tokens_to_tensor<B: Backend>(tokens: &[i64], device: &B::Device) -> Tensor<B, 2, Int> {
    let shape = [1, tokens.len()];
    let data = TensorData::new(tokens.to_vec(), shape);
    Tensor::<B, 2, Int>::from_data(data, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_params_defaults() {
        let params = SamplingParams::default();
        assert!((params.temperature - 1.0).abs() < 1e-5);
        assert!((params.min_p - 0.0).abs() < 1e-5);
        assert_eq!(params.top_k, 0);
        assert!((params.top_p - 1.0).abs() < 1e-5);
        assert!((params.repetition_penalty - 1.0).abs() < 1e-5);
    }
}
