use anyhow::Result;
use burn::backend::{Autodiff, Wgpu};
use burn_llama::{train, TrainingConfig};
use clap::Parser;
use log::{info, LevelFilter};

/// Minimal launcher: `cargo run --release -- configs/test.yaml`
#[derive(Parser, Debug)]
#[command(name = "burn-llama")]
#[command(about = "Train decoder experiments from a YAML config", long_about = None)]
struct Cli {
    /// Path to YAML training config (defaults to configs/test.yaml)
    #[arg(value_name = "CONFIG", default_value = "configs/test.yaml")]
    config: String,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

// Use Wgpu backend with autodiff for training
type Backend = Wgpu;
type AutodiffBackend = Autodiff<Backend>;

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match cli.verbose {
        0 => LevelFilter::Info,
        1 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };
    env_logger::Builder::new()
        .filter_level(LevelFilter::Warn)
        .filter_module("burn_llama", log_level)
        .init();

    // Get device
    let device = burn::backend::wgpu::WgpuDevice::default();
    info!("Using device: {:?}", device);

    let training_config = if cli.config.ends_with(".yaml") || cli.config.ends_with(".yml") {
        info!("Loading config from: {}", cli.config);
        load_config(&cli.config)?
    } else {
        info!(
            "Config '{}' not recognized as YAML, falling back to built-in preset '{}'",
            cli.config, cli.config
        );
        match cli.config.as_str() {
            "test" => TrainingConfig::test(),
            "nano" => TrainingConfig::nano(),
            "small" => TrainingConfig::small(),
            _ => anyhow::bail!("Provide a YAML file path or one of the presets: test|nano|small"),
        }
    };

    info!("Starting training with config: {:#?}", training_config);
    train::<AutodiffBackend>(training_config, device)?;

    Ok(())
}

fn load_config(path: &str) -> Result<TrainingConfig> {
    let config_str = std::fs::read_to_string(path)?;
    let config: TrainingConfig = serde_yaml::from_str(&config_str)?;
    Ok(config)
}
