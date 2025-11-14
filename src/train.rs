use anyhow::Result;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer},
    prelude::*,
    tensor::{activation::softmax, backend::AutodiffBackend, ElementConversion, Int, Tensor},
};
use std::time::Instant;

use crate::{
    config::TrainingConfig,
    data::{CharDataset, TextBatcher},
    models::LlamaModel,
};

/// Modern training function using Burn 0.19 APIs
pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) -> Result<()> {
    // Set random seed for reproducibility
    B::seed(&device, config.seed);

    // Initialize model
    let mut model = LlamaModel::<B>::new(config.model.clone(), &device);
    log::info!(
        "Initialized model with {} parameters",
        format_num_params(&model)
    );

    // Initialize optimizer
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Value(config.gradient_clip)))
        .init();

    // Create datasets
    let train_dataset = CharDataset::from_file(&config.train_data)?;
    let val_dataset = CharDataset::from_file(&config.val_data)?;

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

    // Loss function
    let loss_fn = CrossEntropyLossConfig::new()
        .with_weights(None)
        .with_smoothing(Some(0.1)) // Label smoothing
        .init(&device);

    // Training metrics
    let mut global_step = 0;
    let mut best_val_loss = f32::INFINITY;
    let start_time = Instant::now();

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
        };
        model = train_epoch(model, &mut optimizer, epoch_ctx)?;

        // Validation phase
        if epoch % config.val_frequency == 0 {
            let val_loss = validate(&model, dataloader_val.as_ref(), &loss_fn, epoch, &device)?;

            // Save checkpoint if best model
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                save_checkpoint(&model, epoch, val_loss)?;
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

    for batch in ctx.dataloader.iter() {
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
    }

    Ok(model)
}

/// Validation loop
fn validate<B: AutodiffBackend>(
    model: &LlamaModel<B>,
    dataloader: &dyn DataLoader<B, TextBatch<B>>,
    loss_fn: &CrossEntropyLoss<B>,
    epoch: usize,
    device: &B::Device,
) -> Result<f32> {
    let mut total_loss = 0.0f32;
    let mut num_batches = 0;

    for batch in dataloader.iter() {
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

    let avg_loss = total_loss / num_batches as f32;
    log::info!("[Valid] Epoch: {} | Loss: {:.4}", epoch, avg_loss);

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
    _model: &LlamaModel<B>,
    epoch: usize,
    val_loss: f32,
) -> Result<()> {
    let checkpoint_name = format!("checkpoint_epoch_{}_loss_{:.4}.bin", epoch, val_loss);
    log::info!("Saving checkpoint: {}", checkpoint_name);

    // In real implementation, use burn::record::Recorder
    // model.save_file(checkpoint_name, &CompactRecorder::new())?;

    Ok(())
}

/// Format number of parameters
fn format_num_params<B: AutodiffBackend, M: AutodiffModule<B>>(model: &M) -> String {
    let num_params = model.num_params();

    if num_params > 1_000_000 {
        format!("{:.2}M", num_params as f32 / 1_000_000.0)
    } else if num_params > 1_000 {
        format!("{:.2}K", num_params as f32 / 1_000.0)
    } else {
        format!("{}", num_params)
    }
}

/// Batch structure for text data
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
}
