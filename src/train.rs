use anyhow::{anyhow, Result};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::{backend::AutodiffBackend, DType, ElementConversion, Int, Tensor},
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    fs,
    time::{Duration, Instant},
};

use crate::{
    config::{ModelConfig, TrainingConfig},
    data::{ByteTokenizer, CharDataset, TextBatcher, Tokenizer, WikiDataset},
    models::LlamaModel,
    sampling::{sample_next_token, SamplingParams},
};

/// Training loop aligned with `train.py`.
pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) -> Result<()> {
    B::seed(&device, config.seed);

    use crate::models::llama::MpPolicy;
    // MpPolicy controls per-layer GEMM precision for experimentation.
    // When mixed_precision is enabled and BF16 is supported, prefer BF16 GEMMs.
    let mp_policy = if config.mixed_precision && B::supports_dtype(&device, DType::BF16) {
        MpPolicy::bf16()
    } else {
        if config.mixed_precision {
            log::warn!("mixed_precision=true but BF16 GEMMs are unsupported; falling back to fp32");
        }
        MpPolicy::fp32()
    };
    let mut model = LlamaModel::<B>::new(config.model.clone(), &device, mp_policy);
    log::info!(
        "Total parameters: {}",
        format_param_count(count_params(&model))
    );

    let weight_decay = if config.weight_decay > 0.0 {
        Some(WeightDecayConfig::new(config.weight_decay))
    } else {
        None
    };
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Value(config.gradient_clip)))
        .init();

    let (train_dataset, val_dataset) = load_datasets(&config)?;
    let batcher = TextBatcher::new(config.sequence_length);
    let tokenizer = ByteTokenizer;

    let total_steps = config.num_batches.max(1);
    let mut global_step = 0usize;
    let mut best_val_loss = f32::INFINITY;

    let progress = ProgressBar::new(total_steps as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "training: {elapsed_precise} |{bar:40.cyan/blue}| {pos}/{len} ({eta_precise}) [{msg}]",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    progress.enable_steady_tick(Duration::from_millis(200));
    progress.set_message("loss=---- | it/s=--");
    let train_start = Instant::now();

    // Initial validation
    let val_start = validate(
        &model.valid(),
        &val_dataset,
        &batcher,
        config.batch_size,
        config.val_batches,
        &device,
    )?;
    log::info!("Step {} | Val loss: {:.4}", global_step, val_start);

    if config.batch_size == 0 {
        log::warn!("batch_size of 0 is invalid; defaulting to 1 for training");
    }
    if config.gradient_accumulation_steps == 0 {
        log::warn!("gradient_accumulation_steps of 0 is invalid; defaulting to 1");
    }
    while global_step < total_steps {
        let mut loss_tensor: Option<Tensor<B, 1>> = None;
        let mut token_sum = 0.0f32;
        let mut saved_checkpoint = false;
        let micro_batch_size = config.batch_size.max(1);
        let grad_accum_steps = config.gradient_accumulation_steps.max(1);

        for _ in 0..grad_accum_steps {
            let batch = train_dataset.sample_batch::<B>(&batcher, micro_batch_size, &device);

            let logits = model.forward(batch.tokens.clone(), 0);
            let [batch_size, seq_len, vocab_size] = logits.dims();
            let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
            let targets_flat = batch.targets.reshape([batch_size * seq_len]);

            let loss = cross_entropy(logits_flat, targets_flat);
            let tokens = (batch_size * seq_len) as f32;
            let loss_sum_tensor = loss.clone() * tokens;

            loss_tensor = Some(match loss_tensor {
                Some(acc) => acc + loss_sum_tensor.clone(),
                None => loss_sum_tensor.clone(),
            });

            token_sum += tokens;
        }

        let total_loss_tensor =
            loss_tensor.ok_or_else(|| anyhow!("No loss accumulated for this step"))?;
        let normalized_loss = total_loss_tensor.clone() / token_sum;
        let grads = normalized_loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);

        global_step += 1;
        // Single host sync per optimizer step (not per micro-batch)
        let avg_loss = normalized_loss.into_scalar().elem::<f32>();
        progress.set_position(global_step as u64);
        let elapsed = train_start.elapsed().as_secs_f64().max(1e-9);
        let it_per_sec = global_step as f64 / elapsed;
        progress.set_message(format!("loss={avg_loss:.4} | it/s={it_per_sec:.2}"));

        if config.validate_every > 0 && global_step % config.validate_every == 0 {
            let val_model = model.valid();
            let val_loss = progress.suspend(|| {
                validate(
                    &val_model,
                    &val_dataset,
                    &batcher,
                    config.batch_size,
                    config.val_batches,
                    &device,
                )
            })?;
            log::info!("Step {} | Val loss: {:.4}", global_step, val_loss);
            if val_loss.is_finite() && val_loss < best_val_loss {
                best_val_loss = val_loss;
                save_checkpoint(
                    &model,
                    &config.model,
                    global_step,
                    val_loss,
                    &config.output_dir,
                )?;
                log::info!("New best model saved with validation loss: {:.4}", val_loss);
                saved_checkpoint = true;
            }
        }

        if config.generate_every > 0 && global_step % config.generate_every == 0 {
            let gen_model = model.valid();
            progress.suspend(|| {
                generate_preview(
                    &gen_model,
                    &val_dataset,
                    &tokenizer,
                    &GenerationSettings {
                        prompt_len: config.generation_prompt_length,
                        gen_len: config.generation_length,
                        temperature: config.temperature,
                        min_p: config.min_p,
                    },
                    global_step,
                    &device,
                )
            })?;
        }

        if config.save_every > 0 && global_step % config.save_every == 0 && !saved_checkpoint {
            save_checkpoint(
                &model,
                &config.model,
                global_step,
                best_val_loss,
                &config.output_dir,
            )?;
        }
    }

    progress.finish_with_message("done");
    save_final_checkpoint(&model, &config.model, &config.output_dir)?;

    Ok(())
}

fn validate<B: Backend>(
    model: &LlamaModel<B>,
    dataset: &CharDataset,
    batcher: &TextBatcher,
    batch_size: usize,
    num_batches: usize,
    device: &B::Device,
) -> Result<f32> {
    if num_batches == 0 {
        return Ok(f32::NAN);
    }

    let mut total_loss = 0.0f32;
    let mut token_sum = 0.0f32;

    for _ in 0..num_batches {
        let batch = dataset.sample_batch::<B>(batcher, batch_size, device);
        let logits = model.forward(batch.tokens.clone(), 0);
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        let loss = cross_entropy(logits_flat, targets_flat);
        let tokens = (batch_size * seq_len) as f32;
        total_loss += loss.into_scalar().elem::<f32>() * tokens;
        token_sum += tokens;
    }

    if token_sum == 0.0 {
        Ok(f32::NAN)
    } else {
        Ok(total_loss / token_sum)
    }
}

fn generate_preview<B: Backend>(
    model: &LlamaModel<B>,
    dataset: &CharDataset,
    tokenizer: &ByteTokenizer,
    settings: &GenerationSettings,
    step: usize,
    device: &B::Device,
) -> Result<()> {
    if let Some(prompt_tokens) = dataset.sample_prompt(settings.prompt_len) {
        let prompt_text = tokenizer.decode(&prompt_tokens);
        let generated = generate_text(
            model,
            &prompt_tokens,
            settings.gen_len,
            settings.temperature,
            settings.min_p,
            device,
        )?;
        let completion = if generated.len() > prompt_tokens.len() {
            &generated[prompt_tokens.len()..]
        } else {
            &[]
        };
        let completion_text = tokenizer.decode(completion);

        log::info!(
            "================================================== Step {} ==================================================",
            step
        );
        log::info!("Prompt: {}", prompt_text);
        log::info!("Generated: {}", completion_text);
    }
    Ok(())
}

fn generate_text<B: Backend>(
    model: &LlamaModel<B>,
    prompt: &[i64],
    max_length: usize,
    temperature: f32,
    min_p: f32,
    device: &B::Device,
) -> Result<Vec<i64>> {
    if prompt.is_empty() {
        return Ok(Vec::new());
    }

    let params = SamplingParams::new()
        .with_temperature(temperature)
        .with_min_p(min_p);

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

        let next_token = sample_next_token(last_logits, &params, &tokens)?;
        tokens.push(next_token);
    }

    Ok(tokens)
}

struct GenerationSettings {
    prompt_len: usize,
    gen_len: usize,
    temperature: f32,
    min_p: f32,
}
fn cross_entropy<B: Backend>(logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
    use burn::tensor::DType;
    // Upcast to fp32 for log_softmax to avoid precision issues with bf16/f16
    let dtype = logits.dtype();
    let use_fp32 = matches!(dtype, DType::BF16 | DType::F16);
    let logits_compute = if use_fp32 {
        logits.cast(DType::F32)
    } else {
        logits
    };
    let log_probs = burn::tensor::activation::log_softmax(logits_compute, 1);
    let [batch, _] = log_probs.dims();
    let gathered = log_probs
        .gather(1, targets.reshape([batch, 1]))
        .reshape([batch]);
    let loss = gathered.mean().neg();
    // Cast loss back to original dtype for gradient computation compatibility
    if use_fp32 {
        loss.cast(dtype)
    } else {
        loss
    }
}

fn tokens_to_tensor<B: Backend>(tokens: &[i64], device: &B::Device) -> Tensor<B, 2, Int> {
    let shape = [1, tokens.len()];
    let data = TensorData::new(tokens.to_vec(), shape);
    Tensor::<B, 2, Int>::from_data(data, device)
}

fn save_checkpoint<B: AutodiffBackend>(
    model: &LlamaModel<B>,
    model_config: &ModelConfig,
    step: usize,
    val_loss: f32,
    output_dir: &str,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let checkpoint_name = format!(
        "{}/checkpoint_step_{}_loss_{:.4}.bin",
        output_dir, step, val_loss
    );
    log::info!("Saving checkpoint: {}", checkpoint_name);
    CompactRecorder::new()
        .record(model.valid().into_record(), checkpoint_name.clone().into())
        .map_err(|err| anyhow!(err.to_string()))?;

    // Save model config alongside checkpoint
    let config_path = checkpoint_name.replace(".bin", ".config.json");
    let config_json = serde_json::to_string_pretty(model_config)?;
    fs::write(&config_path, config_json)?;
    log::debug!("Model config saved to: {}", config_path);

    Ok(())
}

fn save_final_checkpoint<B: AutodiffBackend>(
    model: &LlamaModel<B>,
    model_config: &ModelConfig,
    output_dir: &str,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let path = format!("{}/final.bin", output_dir);
    let record = model.valid().into_record();
    CompactRecorder::new()
        .record(record, path.clone().into())
        .map_err(|err| anyhow!(err.to_string()))?;

    // Save model config alongside checkpoint
    let config_path = format!("{}/final.config.json", output_dir);
    let config_json = serde_json::to_string_pretty(model_config)?;
    fs::write(&config_path, &config_json)?;
    log::info!("Model config saved to: {}", config_path);

    log::info!("Training complete! Final checkpoint saved to {}", path);
    Ok(())
}

fn load_datasets(config: &TrainingConfig) -> Result<(CharDataset, CharDataset)> {
    if config.train_data.ends_with(".gz") {
        let wiki = WikiDataset::load_enwik8(&config.train_data)?;
        let train = wiki.train_dataset(config.sequence_length);
        let val = wiki.val_dataset(config.sequence_length);
        Ok((train, val))
    } else {
        let train = CharDataset::from_file(&config.train_data)?
            .with_sequence_length(config.sequence_length);
        let val =
            CharDataset::from_file(&config.val_data)?.with_sequence_length(config.sequence_length);
        Ok((train, val))
    }
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> TextBatch<B> {
    pub fn to_device(self, device: &B::Device) -> Self {
        Self {
            tokens: self.tokens.to_device(device),
            targets: self.targets.to_device(device),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.tokens.dims()[0]
    }

    pub fn chunk(&self, index: usize, chunk_size: usize) -> Self {
        let start = index * chunk_size;
        let end = start + chunk_size;
        self.slice_rows(start, end)
    }

    pub fn slice_rows(&self, start: usize, end: usize) -> Self {
        let total = self.batch_size();
        assert!(
            end <= total,
            "requested slice end {} exceeds batch size {}",
            end,
            total
        );
        assert!(start < end, "invalid slice: start {} >= end {}", start, end);
        let seq_len = self.tokens.dims()[1];
        Self {
            tokens: self.tokens.clone().slice([start..end, 0..seq_len]),
            targets: self.targets.clone().slice([start..end, 0..seq_len]),
        }
    }
}

fn count_params<B: AutodiffBackend, M: AutodiffModule<B>>(model: &M) -> usize {
    model.num_params()
}

fn format_param_count(count: usize) -> String {
    if count > 1_000_000 {
        format!("{:.2}M", count as f32 / 1_000_000.0)
    } else if count > 1_000 {
        format!("{:.2}K", count as f32 / 1_000.0)
    } else {
        format!("{}", count)
    }
}
