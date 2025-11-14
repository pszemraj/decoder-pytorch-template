use anyhow::{anyhow, Result};
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer},
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, backend::AutodiffBackend, ElementConversion, Int, Tensor},
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    fs,
    time::{Duration, Instant},
};

use crate::{
    config::{ModelConfig, TrainingConfig},
    data::{CharDataset, TextBatcher, WikiDataset},
    models::LlamaModel,
};

/// Modern training function using Burn 0.19 APIs
pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) -> Result<()> {
    // Set random seed for reproducibility
    B::seed(&device, config.seed);

    // Initialize model
    let mut model = LlamaModel::<B>::new(config.model.clone(), &device);
    let total_params = count_params(&model);
    log_model_summary(&config.model, total_params);
    log::info!(
        "Initialized model with {} parameters",
        format_param_count(total_params)
    );

    // Initialize optimizer
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Value(config.gradient_clip)))
        .init();

    // Create datasets
    let (train_dataset, val_dataset) = load_datasets(&config)?;

    // Create data loaders
    let batcher = TextBatcher::new(config.sequence_length);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_val = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(val_dataset);

    let train_step_limit = steps_limit(config.train_steps_per_epoch);
    let val_step_limit = steps_limit(config.val_steps);
    let progress_total = train_step_limit
        .or_else(|| estimate_total_steps(dataloader_train.as_ref(), config.batch_size));

    // Loss function
    let loss_fn = CrossEntropyLossConfig::new()
        .with_weights(None)
        .with_smoothing(None)
        .init(&device);

    // Training metrics
    let mut global_step = 0;
    let mut best_val_loss = f32::INFINITY;
    let start_time = Instant::now();

    // Initial validation
    validate(
        &model,
        dataloader_val.as_ref(),
        &loss_fn,
        0,
        &device,
        val_step_limit,
        global_step,
    )?;

    // Training loop
    for epoch in 1..=config.num_epochs {
        log::info!("Starting epoch {}/{}", epoch, config.num_epochs);

        // Training phase
        let epoch_ctx = EpochContext {
            dataloader: dataloader_train.as_ref(),
            loss_fn: &loss_fn,
            learning_rate: config.learning_rate,
            gradient_accumulation_steps: config.gradient_accumulation_steps,
            global_step: &mut global_step,
            epoch,
            device: &device,
            max_steps: train_step_limit,
            progress_total,
        };
        model = train_epoch(model, &mut optimizer, epoch_ctx)?;

        // Validation phase
        if epoch % config.val_frequency == 0 {
            let val_loss = validate(
                &model,
                dataloader_val.as_ref(),
                &loss_fn,
                epoch,
                &device,
                val_step_limit,
                global_step,
            )?;

            // Save checkpoint if best model
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                save_checkpoint(&model, epoch, val_loss, &config.output_dir)?;
                log::info!("New best model saved with validation loss: {:.4}", val_loss);
            }
        }

        // Generate samples
        if epoch % config.sample_frequency == 0 {
            generate_samples(&model.valid(), &device)?;
        }

        // Log training progress
        let elapsed = start_time.elapsed();
        log::info!(
            "Epoch {} completed in {:.2}s | Best val loss: {:.4}",
            epoch,
            elapsed.as_secs_f32(),
            best_val_loss
        );
    }

    save_final_checkpoint(&model, &config.output_dir)?;

    Ok(())
}

struct EpochContext<'a, B: AutodiffBackend> {
    dataloader: &'a dyn DataLoader<B, TextBatch<B>>,
    loss_fn: &'a CrossEntropyLoss<B>,
    learning_rate: f64,
    gradient_accumulation_steps: usize,
    global_step: &'a mut usize,
    epoch: usize,
    device: &'a B::Device,
    max_steps: Option<usize>,
    progress_total: Option<usize>,
}

/// Train for one epoch with gradient accumulation
fn train_epoch<'a, B: AutodiffBackend>(
    mut model: LlamaModel<B>,
    optimizer: &mut impl Optimizer<LlamaModel<B>, B>,
    ctx: EpochContext<'a, B>,
) -> Result<LlamaModel<B>> {
    assert!(
        ctx.gradient_accumulation_steps > 0,
        "gradient_accumulation_steps must be > 0"
    );
    let mut accumulated_loss = 0.0f32;
    let mut accumulation_count = 0usize;
    let mut accumulator = GradientsAccumulator::<LlamaModel<B>>::new();
    let mut loss_sum = 0.0f32;
    let mut token_sum = 0.0f32;
    let progress = create_progress_bar(ctx.epoch, ctx.progress_total);
    let use_spinner = ctx.progress_total.is_none();

    for (step, batch) in ctx.dataloader.iter().enumerate() {
        if let Some(limit) = ctx.max_steps {
            if step >= limit {
                break;
            }
        }
        // Move batch to device
        let batch = batch.to_device(ctx.device);

        // Forward pass
        let logits = model.forward(batch.tokens.clone(), 0);

        // Reshape for loss calculation
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        // Calculate loss
        let loss = ctx.loss_fn.forward(logits_flat, targets_flat);
        let loss_value = loss.clone().into_scalar().elem::<f32>();
        accumulated_loss += loss_value;
        accumulation_count += 1;
        let tokens = (batch_size * seq_len) as f32;
        loss_sum += loss_value * tokens;
        token_sum += tokens;

        // Scale loss for gradient accumulation
        let scaled_loss = loss / ctx.gradient_accumulation_steps as f32;

        // Backward pass
        let grads = GradientsParams::from_grads(scaled_loss.backward(), &model);
        accumulator.accumulate(&model, grads);

        // Optimizer step after accumulation
        if accumulation_count == ctx.gradient_accumulation_steps {
            let grads = accumulator.grads();
            model = optimizer.step(ctx.learning_rate, model, grads);

            *ctx.global_step += 1;
            if *ctx.global_step % 10 == 0 {
                let avg_loss = accumulated_loss / ctx.gradient_accumulation_steps as f32;
                log::info!(
                    "[Train] Epoch: {} | Step: {} | Loss: {:.4}",
                    ctx.epoch,
                    *ctx.global_step,
                    avg_loss
                );
            }

            accumulated_loss = 0.0;
            accumulation_count = 0;
            let avg_loss = if token_sum > 0.0 {
                loss_sum / token_sum
            } else {
                0.0
            };
            if ctx.progress_total.is_some() {
                progress.inc(1);
            } else {
                progress.tick();
            }
            progress.set_message(format!("loss={avg_loss:.4}"));
            loss_sum = 0.0;
            token_sum = 0.0;
        } else if use_spinner {
            progress.tick();
        }
    }

    // Handle remaining gradients
    if accumulation_count > 0 {
        let grads = accumulator.grads();
        model = optimizer.step(ctx.learning_rate, model, grads);
        *ctx.global_step += 1;
        let avg_loss = accumulated_loss / accumulation_count as f32;
        log::info!(
            "[Train] Epoch: {} | Step: {} | Loss: {:.4}",
            ctx.epoch,
            *ctx.global_step,
            avg_loss
        );
        if ctx.progress_total.is_some() {
            progress.inc(1);
        } else {
            progress.tick();
        }
        progress.set_message(format!("loss={avg_loss:.4}"));
    }

    if ctx.progress_total.is_some() {
        progress.finish();
    } else {
        progress.finish_and_clear();
    }

    Ok(model)
}

/// Validation loop
fn validate<B: Backend>(
    model: &LlamaModel<B>,
    dataloader: &dyn DataLoader<B, TextBatch<B>>,
    loss_fn: &CrossEntropyLoss<B>,
    epoch: usize,
    device: &B::Device,
    max_steps: Option<usize>,
    global_step: usize,
) -> Result<f32> {
    let mut total_loss = 0.0f32;
    let mut num_batches = 0;

    for (step, batch) in dataloader.iter().enumerate() {
        if let Some(limit) = max_steps {
            if step >= limit {
                break;
            }
        }
        let batch = batch.to_device(device);

        // Forward pass (no gradients needed)
        let logits = model.forward(batch.tokens.clone(), 0);

        // Calculate loss
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        let loss = loss_fn.forward(logits_flat, targets_flat);
        total_loss += loss.into_scalar().elem::<f32>();
        num_batches += 1;
    }

    if num_batches == 0 {
        log::warn!(
            "Validation dataloader produced no batches at epoch {}. Skipping metric.",
            epoch
        );
        return Ok(f32::NAN);
    }

    let avg_loss = total_loss / num_batches as f32;
    log::info!(
        "Step {} | Epoch {} | Val loss: {:.4}",
        global_step,
        epoch,
        avg_loss
    );

    Ok(avg_loss)
}

/// Generate text samples
fn generate_samples<B: Backend>(model: &LlamaModel<B>, device: &B::Device) -> Result<()> {
    let prompts = vec![
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
    ];

    for prompt in prompts {
        let tokens = tokenize(prompt);
        if tokens.is_empty() {
            continue;
        }

        let generated = generate_text(model, tokens, 50, 1.0, device)?;
        let text = detokenize(&generated);

        log::info!("Generated from '{}': {}", prompt, text);
    }

    Ok(())
}

/// Simple character tokenization
fn tokenize(text: &str) -> Vec<i64> {
    text.bytes().map(|b| b as i64).collect()
}

/// Simple character detokenization  
fn detokenize(tokens: &[i64]) -> String {
    tokens.iter().map(|&t| (t as u8) as char).collect()
}

fn tokens_to_tensor<B: Backend>(tokens: &[i64], device: &B::Device) -> Tensor<B, 2, Int> {
    let shape = [1, tokens.len()];
    let data = TensorData::new(tokens.to_vec(), shape);
    Tensor::<B, 2, Int>::from_data(data, device)
}

/// Generate text using the model
fn generate_text<B: Backend>(
    model: &LlamaModel<B>,
    prompt_tokens: Vec<i64>,
    max_length: usize,
    temperature: f32,
    device: &B::Device,
) -> Result<Vec<i64>> {
    if prompt_tokens.is_empty() {
        return Ok(prompt_tokens);
    }

    let mut tokens = prompt_tokens;
    let max_context = model.max_position_embeddings();
    let temp = temperature.max(1e-5);

    for _ in 0..max_length {
        let start = tokens.len().saturating_sub(max_context);
        let context = tokens[start..].to_vec();
        if context.is_empty() {
            break;
        }

        let position_offset = start;
        let input = tokens_to_tensor::<B>(&context, device);
        let logits = model.forward(input, position_offset);

        let last_index = context.len() - 1;
        let vocab = logits.dims()[2];
        let last_logits = logits
            .slice([0..1, last_index..last_index + 1, 0..vocab])
            .squeeze_dims(&[0, 1]);

        let probs = softmax(last_logits / temp, 0);
        let next_token = sample_from_probs(probs)?;
        tokens.push(next_token);
    }

    Ok(tokens)
}

/// Sample from probability distribution
fn sample_from_probs<B: Backend>(probs: Tensor<B, 1>) -> Result<i64> {
    let values: Vec<f32> = probs.into_data().iter::<f32>().collect();
    let mut cumsum = 0.0;
    let random = rand::random::<f32>();

    for (idx, prob) in values.iter().enumerate() {
        cumsum += prob;
        if cumsum > random {
            return Ok(idx as i64);
        }
    }

    Ok((values.len().saturating_sub(1)) as i64)
}

/// Save model checkpoint
fn save_checkpoint<B: AutodiffBackend>(
    model: &LlamaModel<B>,
    epoch: usize,
    val_loss: f32,
    output_dir: &str,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let checkpoint_name = format!(
        "{}/checkpoint_epoch_{}_loss_{:.4}.bin",
        output_dir, epoch, val_loss
    );
    log::info!("Saving checkpoint: {}", checkpoint_name);
    CompactRecorder::new()
        .record(model.valid().into_record(), checkpoint_name.clone().into())
        .map_err(|err| anyhow!(err.to_string()))?;

    Ok(())
}

fn save_final_checkpoint<B: AutodiffBackend>(
    model: &LlamaModel<B>,
    output_dir: &str,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let path = format!("{}/final.bin", output_dir);
    let record = model.valid().into_record();
    CompactRecorder::new()
        .record(record, path.clone().into())
        .map_err(|err| anyhow!(err.to_string()))?;
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

fn steps_limit(value: usize) -> Option<usize> {
    if value == 0 {
        None
    } else {
        Some(value)
    }
}

fn estimate_total_steps<B: Backend>(
    dataloader: &dyn DataLoader<B, TextBatch<B>>,
    batch_size: usize,
) -> Option<usize> {
    if batch_size == 0 {
        return None;
    }
    let items = dataloader.num_items();
    if items == 0 {
        None
    } else {
        Some(items.div_ceil(batch_size))
    }
}

/// Batch structure for text data
#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
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

fn log_model_summary(config: &ModelConfig, total_params: usize) {
    let embed_params = config.vocab_size * config.hidden_size;
    let attn_params = 4 * config.hidden_size * config.hidden_size;
    let ffn_params = 3 * config.hidden_size * config.intermediate_size;
    let norm_params = 2 * config.hidden_size;
    let block_params = attn_params + ffn_params + norm_params;
    let head_params = if config.tie_embeddings {
        0
    } else {
        config.hidden_size * config.vocab_size
    };

    log::info!("======================================================");
    log::info!("Layer (type)          Param Shape  Param #    Grad State");
    log::info!(
        "  Embedding             {:>5}x{:<5} {:>10}   trainable",
        config.vocab_size,
        config.hidden_size,
        embed_params
    );
    log::info!(
        "  TransformerBlock x{:>2}        {:>10}   mixed",
        config.n_layers,
        block_params
    );
    log::info!(
        "  RMSNorm (final)                  {:>10}   trainable",
        config.hidden_size
    );
    if !config.tie_embeddings {
        log::info!(
            "  LM Head               {:>5}x{:<5} {:>10}   trainable",
            config.hidden_size,
            config.vocab_size,
            head_params
        );
    } else {
        log::info!("  LM Head (tied)                    uses embedding weights");
    }
    log::info!("======================================================");
    log::info!("Total params: {}", format_param_count(total_params));
    log::info!("Trainable params: {}", format_param_count(total_params));
    log::info!("Non-trainable params: 0");
    log::info!("======================================================");
}

fn create_progress_bar(epoch: usize, total: Option<usize>) -> ProgressBar {
    let prefix = format!("Epoch {}", epoch);
    match total {
        Some(len) if len > 0 => {
            let pb = ProgressBar::new(len as u64);
            pb.set_style(
                ProgressStyle::with_template("{prefix:<10} {wide_bar} {pos}/{len} [{msg}]")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            pb.set_prefix(prefix);
            pb
        }
        _ => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template("{prefix:<10} {spinner} {msg}")
                    .unwrap()
                    .tick_chars("/-\\| "),
            );
            pb.set_prefix(prefix);
            pb.enable_steady_tick(Duration::from_millis(100));
            pb
        }
    }
}

impl<B: Backend> TextBatch<B> {
    pub fn to_device(self, device: &B::Device) -> Self {
        Self {
            tokens: self.tokens.to_device(device),
            targets: self.targets.to_device(device),
        }
    }
}
