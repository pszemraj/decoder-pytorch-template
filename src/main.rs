use anyhow::Result;
use burn::backend::{Autodiff, Wgpu};
use burn_llama::{train, TrainingConfig};
use clap::{Parser, Subcommand};
use log::info;

/// Modern Llama implementation in Burn 0.19
#[derive(Parser, Debug)]
#[command(name = "burn-llama")]
#[command(about = "Train and run Llama models with Burn", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train a new model
    Train {
        /// Configuration preset (test, nano, small)
        #[arg(short, long, default_value = "nano")]
        preset: String,

        /// Path to custom config file (overrides preset)
        #[arg(short, long)]
        config: Option<String>,

        /// Override batch size
        #[arg(long)]
        batch_size: Option<usize>,

        /// Override learning rate
        #[arg(long)]
        learning_rate: Option<f64>,

        /// Override number of epochs
        #[arg(long)]
        epochs: Option<usize>,
    },

    /// Generate text from a trained model
    Generate {
        /// Path to model checkpoint
        #[arg(short, long)]
        checkpoint: String,

        /// Input prompt
        #[arg(short, long, default_value = "Once upon a time")]
        prompt: String,

        /// Maximum generation length
        #[arg(long, default_value = "100")]
        max_length: usize,

        /// Temperature for sampling
        #[arg(long, default_value = "0.8")]
        temperature: f32,
    },

    /// Evaluate a model on validation data
    Evaluate {
        /// Path to model checkpoint
        #[arg(short, long)]
        checkpoint: String,

        /// Path to validation data
        #[arg(long)]
        data: String,
    },
}

// Use Wgpu backend with autodiff for training
type Backend = Wgpu;
type AutodiffBackend = Autodiff<Backend>;

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match cli.verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    // Get device
    let device = burn::backend::wgpu::WgpuDevice::default();
    info!("Using device: {:?}", device);

    match cli.command {
        Commands::Train {
            preset,
            config,
            batch_size,
            learning_rate,
            epochs,
        } => {
            let mut training_config = if let Some(config_path) = config {
                // Load custom config
                info!("Loading config from: {}", config_path);
                load_config(&config_path)?
            } else {
                // Use preset
                info!("Using preset: {}", preset);
                match preset.as_str() {
                    "test" => TrainingConfig::test(),
                    "nano" => TrainingConfig::nano(),
                    "small" => TrainingConfig::small(),
                    _ => {
                        anyhow::bail!("Unknown preset: {}. Use 'test', 'nano', or 'small'", preset);
                    }
                }
            };

            // Apply overrides
            if let Some(bs) = batch_size {
                training_config = training_config.with_batch_size(bs);
                info!("Overriding batch size to: {}", bs);
            }
            if let Some(lr) = learning_rate {
                training_config = training_config.with_learning_rate(lr);
                info!("Overriding learning rate to: {}", lr);
            }
            if let Some(e) = epochs {
                training_config = training_config.with_num_epochs(e);
                info!("Overriding epochs to: {}", e);
            }

            info!("Starting training with config: {:#?}", training_config);
            train::<AutodiffBackend>(training_config, device)?;
        }

        Commands::Generate {
            checkpoint,
            prompt,
            max_length,
            temperature,
        } => {
            info!("Loading model from: {}", checkpoint);
            info!("Generating text from prompt: '{}'", prompt);
            info!("Max length: {}, Temperature: {}", max_length, temperature);

            // Load model and generate
            generate_text(checkpoint, prompt, max_length, temperature, device)?;
        }

        Commands::Evaluate { checkpoint, data } => {
            info!("Evaluating model: {}", checkpoint);
            info!("Using data: {}", data);

            // Evaluate model
            evaluate_model(checkpoint, data, device)?;
        }
    }

    Ok(())
}

fn load_config(path: &str) -> Result<TrainingConfig> {
    let config_str = std::fs::read_to_string(path)?;
    let config: TrainingConfig = serde_yaml::from_str(&config_str)?;
    Ok(config)
}

fn generate_text(
    _checkpoint: String,
    prompt: String,
    max_length: usize,
    temperature: f32,
    _device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    // TODO: Implement model loading and generation
    info!(
        "Generating {} tokens from '{}' with temperature {}",
        max_length, prompt, temperature
    );

    // Placeholder implementation
    println!("\n=== Generated Text ===");
    println!("{} [generation would continue here...]", prompt);
    println!("===================\n");

    Ok(())
}

fn evaluate_model(
    _checkpoint: String,
    _data: String,
    _device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    // TODO: Implement model evaluation
    info!("Model evaluation not yet implemented");

    Ok(())
}
