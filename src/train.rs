use anyhow::Result;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{backend::AutodiffBackend, Int, Tensor},
};
use std::time::Instant;

use crate::{
    config::TrainingConfig,
    data::{CharDataset, TextBatcher},
    model::LlamaModel,
};

/// Modern training function using Burn 0.19 APIs
pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    device: B::Device,
) -> Result<()> {
    // Set random seed for reproducibility
    B::seed(config.seed);

    // Initialize model
    let mut model = LlamaModel::<B>::new(config.model.clone(), &device);
    log::info!(
        "Initialized model with {} parameters",
        format_num_params(&model)
    );

    // Initialize optimizer
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .init();

    // Initialize gradient clipping
    let gradient_clipping = GradientClippingConfig::Value(config.gradient_clip);

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
        .with_smoothing(Some(0.1))  // Label smoothing
        .init(&device);

    // Training metrics
    let mut global_step = 0;
    let mut best_val_loss = f32::INFINITY;
    let start_time = Instant::now();

    // Training loop
    for epoch in 1..=config.num_epochs {
        log::info!("Starting epoch {}/{}", epoch, config.num_epochs);
        
        // Training phase
        model = train_epoch(
            model,
            &mut optimizer,
            &dataloader_train,
            &loss_fn,
            &gradient_clipping,
            config.learning_rate,
            config.gradient_accumulation_steps,
            &mut global_step,
            epoch,
            &device,
        )?;

        // Validation phase
        if epoch % config.val_frequency == 0 {
            let val_loss = validate(
                &model.valid(),
                &dataloader_val,
                &loss_fn,
                epoch,
                &device,
            )?;

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

/// Train for one epoch with gradient accumulation
fn train_epoch<B: AutodiffBackend>(
    mut model: LlamaModel<B>,
    optimizer: &mut impl Optimizer<LlamaModel<B>, B>,
    dataloader: &dyn Dataset<TextBatch<B>>,
    loss_fn: &CrossEntropyLoss<B>,
    gradient_clipping: &GradientClippingConfig,
    learning_rate: f64,
    gradient_accumulation_steps: usize,
    global_step: &mut usize,
    epoch: usize,
    device: &B::Device,
) -> Result<LlamaModel<B>> {
    let mut accumulated_loss = 0.0;
    let mut accumulated_grads = None;

    for (batch_idx, batch) in dataloader.iter().enumerate() {
        // Move batch to device
        let batch = batch.to_device(device);
        
        // Forward pass
        let logits = model.forward(batch.tokens.clone(), None);
        
        // Reshape for loss calculation
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);
        
        // Calculate loss
        let loss = loss_fn.forward(logits_flat, targets_flat);
        let loss_value = loss.clone().into_scalar();
        accumulated_loss += loss_value;
        
        // Scale loss for gradient accumulation
        let scaled_loss = loss / gradient_accumulation_steps as f32;
        
        // Backward pass
        let grads = scaled_loss.backward();
        
        // Accumulate gradients
        accumulated_grads = match accumulated_grads {
            Some(mut acc_grads) => {
                acc_grads.accumulate(&model, grads);
                Some(acc_grads)
            }
            None => Some(GradientsParams::from_grads(grads, &model)),
        };
        
        // Optimizer step after accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0 {
            if let Some(grads) = accumulated_grads.take() {
                // Apply gradient clipping
                let grads = gradient_clipping.clip_gradients(grads, &model);
                
                // Update model parameters
                model = optimizer.step(learning_rate, model, grads);
                
                // Log metrics
                *global_step += 1;
                if *global_step % 10 == 0 {
                    let avg_loss = accumulated_loss / gradient_accumulation_steps as f64;
                    log::info!(
                        "[Train] Epoch: {} | Step: {} | Loss: {:.4}",
                        epoch, global_step, avg_loss
                    );
                    accumulated_loss = 0.0;
                }
            }
        }
    }
    
    // Handle remaining gradients
    if let Some(grads) = accumulated_grads {
        let grads = gradient_clipping.clip_gradients(grads, &model);
        model = optimizer.step(learning_rate, model, grads);
    }
    
    Ok(model)
}

/// Validation loop
fn validate<B: Backend>(
    model: &LlamaModel<B>,
    dataloader: &dyn Dataset<TextBatch<B>>,
    loss_fn: &CrossEntropyLoss<B>,
    epoch: usize,
    device: &B::Device,
) -> Result<f32> {
    let mut total_loss = 0.0;
    let mut num_batches = 0;

    for batch in dataloader.iter() {
        let batch = batch.to_device(device);
        
        // Forward pass (no gradients needed)
        let logits = model.forward(batch.tokens.clone(), None);
        
        // Calculate loss
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);
        
        let loss = loss_fn.forward(logits_flat, targets_flat);
        total_loss += loss.into_scalar();
        num_batches += 1;
    }

    let avg_loss = total_loss / num_batches as f32;
    log::info!("[Valid] Epoch: {} | Loss: {:.4}", epoch, avg_loss);
    
    Ok(avg_loss)
}

/// Generate text samples
fn generate_samples<B: Backend>(
    model: &LlamaModel<B>,
    device: &B::Device,
) -> Result<()> {
    let prompts = vec![
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
    ];

    for prompt in prompts {
        let tokens = tokenize(prompt);
        let input = Tensor::<B, 2, Int>::from_data(
            tokens.clone(),
            device,
        ).unsqueeze();
        
        let generated = generate_text(model, input, 50, 1.0, device)?;
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
    tokens
        .iter()
        .map(|&t| (t as u8) as char)
        .collect()
}

/// Generate text using the model
fn generate_text<B: Backend>(
    model: &LlamaModel<B>,
    prompt: Tensor<B, 2, Int>,
    max_length: usize,
    temperature: f32,
    device: &B::Device,
) -> Result<Vec<i64>> {
    let mut tokens = prompt.clone().squeeze(0).to_data().to_vec()?;
    
    for _ in 0..max_length {
        let input = Tensor::<B, 2, Int>::from_data(tokens.clone(), device).unsqueeze();
        let logits = model.forward(input, None);
        
        // Get last token logits
        let last_logits = logits
            .slice([0..1, tokens.len() - 1..tokens.len(), ..])
            .squeeze_dims(&[0, 1]);
        
        // Apply temperature
        let probs = softmax(last_logits / temperature, 0);
        
        // Sample next token
        let next_token = sample_from_probs(probs, device)?;
        tokens.push(next_token);
    }
    
    Ok(tokens)
}

/// Sample from probability distribution
fn sample_from_probs<B: Backend>(
    probs: Tensor<B, 1>,
    device: &B::Device,
) -> Result<i64> {
    // Convert to cumulative distribution
    let probs_data = probs.to_data().to_vec::<f32>()?;
    let mut cumsum = 0.0;
    let random = rand::random::<f32>();
    
    for (idx, &prob) in probs_data.iter().enumerate() {
        cumsum += prob;
        if cumsum > random {
            return Ok(idx as i64);
        }
    }
    
    Ok((probs_data.len() - 1) as i64)
}

/// Save model checkpoint
fn save_checkpoint<B: AutodiffBackend>(
    model: &LlamaModel<B>,
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
    let num_params = model
        .num_params()
        .values()
        .map(|p| p.num_elements())
        .sum::<usize>();
    
    if num_params > 1_000_000 {
        format!("{:.2}M", num_params as f32 / 1_000_000.0)
    } else if num_params > 1_000 {
        format!("{:.2}K", num_params as f32 / 1_000.0)
    } else {
        format!("{}", num_params)
    }
}

/// Batch structure for text data
#[derive(Clone)]
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
