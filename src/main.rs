#![recursion_limit = "256"]

use anyhow::Result;
use burn::backend::Autodiff;
use burn_llama::{
    infer::{generate, load_checkpoint, load_model_config},
    sampling::SamplingParams,
    train, TrainingConfig,
};
use clap::{Parser, Subcommand, ValueEnum};
use half::bf16;
use log::{info, LevelFilter};
use serde_saphyr as yaml;
use std::io::Write;
use std::path::PathBuf;

/// Decoder-only transformer training and inference
#[derive(Parser, Debug)]
#[command(name = "burn-llama")]
#[command(about = "Train and run decoder experiments", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train a model from a YAML config
    Train {
        /// Path to YAML training config (or preset name: test, nano, small)
        #[arg(value_name = "CONFIG", default_value = "configs/test.yaml")]
        config: String,

        /// Backend to use (wgpu, cuda, cpu)
        #[arg(long, value_enum)]
        backend: Option<BackendCli>,

        /// Numeric precision (fp32, bf16)
        #[arg(long, value_enum)]
        precision: Option<PrecisionCli>,
    },

    /// Generate text from a trained checkpoint
    Infer {
        /// Path to checkpoint file (.bin)
        #[arg(short, long, required = true)]
        checkpoint: PathBuf,

        /// Text prompt to continue
        #[arg(short, long, required = true)]
        prompt: String,

        /// Optional YAML config for model architecture (for legacy checkpoints without .config.json)
        #[arg(long)]
        config: Option<PathBuf>,

        /// Maximum number of tokens to generate
        #[arg(long, default_value = "100")]
        max_length: usize,

        /// Temperature for sampling (higher = more random)
        #[arg(long, default_value = "0.8")]
        temperature: f32,

        /// Top-k filtering (0 = disabled)
        #[arg(long, default_value = "0")]
        top_k: usize,

        /// Top-p (nucleus) filtering (1.0 = disabled)
        #[arg(long, default_value = "1.0")]
        top_p: f32,

        /// Min-p filtering threshold (0.0 = disabled)
        #[arg(long, default_value = "0.05")]
        min_p: f32,

        /// Repetition penalty (1.0 = no penalty)
        #[arg(long, default_value = "1.0")]
        repetition_penalty: f32,

        /// Backend to use (wgpu, cuda, cpu)
        #[arg(long, value_enum)]
        backend: Option<BackendCli>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendCli {
    Wgpu,
    Cuda,
    Cpu,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PrecisionCli {
    Fp32,
    Bf16,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match cli.verbose {
        0 => LevelFilter::Info,
        1 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };
    env_logger::Builder::new()
        .format(|buf, record| writeln!(buf, "[{}] {}", record.level(), record.args()))
        .filter_level(LevelFilter::Warn)
        .filter_module("burn_llama", log_level)
        .init();

    match cli.command {
        Commands::Train {
            config,
            backend,
            precision,
        } => run_train(config, backend, precision),
        Commands::Infer {
            checkpoint,
            prompt,
            config,
            max_length,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            backend,
        } => run_infer(
            checkpoint,
            prompt,
            config,
            max_length,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            backend,
        ),
    }
}

fn run_train(
    config_path: String,
    backend: Option<BackendCli>,
    precision: Option<PrecisionCli>,
) -> Result<()> {
    let mut training_config = if config_path.ends_with(".yaml") || config_path.ends_with(".yml") {
        info!("Loading config from: {}", config_path);
        load_config(&config_path)?
    } else {
        info!(
            "Config '{}' not recognized as YAML, falling back to built-in preset '{}'",
            config_path, config_path
        );
        match config_path.as_str() {
            "test" => TrainingConfig::test(),
            "nano" => TrainingConfig::nano(),
            "small" => TrainingConfig::small(),
            _ => anyhow::bail!("Provide a YAML file path or one of the presets: test|nano|small"),
        }
    };

    let backend_choice = backend.unwrap_or(BackendCli::Wgpu);
    let precision_choice = precision.unwrap_or(if training_config.mixed_precision {
        PrecisionCli::Bf16
    } else {
        PrecisionCli::Fp32
    });
    if precision.is_some() {
        training_config.mixed_precision = matches!(precision_choice, PrecisionCli::Bf16);
        info!(
            "Overriding mixed_precision to {} based on --precision",
            training_config.mixed_precision
        );
    }

    run_train_backend(backend_choice, precision_choice, training_config)?;

    Ok(())
}

fn load_config(path: &str) -> Result<TrainingConfig> {
    let config_str = std::fs::read_to_string(path)?;
    let config: TrainingConfig = yaml::from_str(&config_str)?;
    Ok(config)
}

fn run_train_backend(
    backend: BackendCli,
    precision: PrecisionCli,
    config: TrainingConfig,
) -> Result<()> {
    match backend {
        BackendCli::Wgpu => run_train_wgpu(precision, config),
        BackendCli::Cuda => run_train_cuda(precision, config),
        BackendCli::Cpu => run_train_cpu(precision, config),
    }
}

#[cfg(feature = "backend-wgpu")]
fn run_train_wgpu(precision: PrecisionCli, config: TrainingConfig) -> Result<()> {
    use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, Wgpu, WgpuDevice};

    let device = WgpuDevice::default();
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    info!("Using WGPU device: {:?}", device);

    match precision {
        PrecisionCli::Fp32 => train::<Autodiff<Wgpu<f32>>>(config, device),
        PrecisionCli::Bf16 => train::<Autodiff<Wgpu<bf16>>>(config, device),
    }
}

#[cfg(not(feature = "backend-wgpu"))]
fn run_train_wgpu(_precision: PrecisionCli, _config: TrainingConfig) -> Result<()> {
    anyhow::bail!("wgpu backend is not enabled. Rebuild with --features backend-wgpu")
}

#[cfg(feature = "backend-cuda")]
fn run_train_cuda(precision: PrecisionCli, config: TrainingConfig) -> Result<()> {
    use burn::backend::cuda::{Cuda, CudaDevice};

    let device = CudaDevice::default();
    info!("Using CUDA device: {:?}", device);

    match precision {
        PrecisionCli::Fp32 => train::<Autodiff<Cuda<f32>>>(config, device),
        PrecisionCli::Bf16 => train::<Autodiff<Cuda<bf16>>>(config, device),
    }
}

#[cfg(not(feature = "backend-cuda"))]
fn run_train_cuda(_precision: PrecisionCli, _config: TrainingConfig) -> Result<()> {
    anyhow::bail!("cuda backend is not enabled. Rebuild with --features backend-cuda")
}

#[cfg(feature = "backend-cpu")]
fn run_train_cpu(precision: PrecisionCli, config: TrainingConfig) -> Result<()> {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    if matches!(precision, PrecisionCli::Bf16) {
        log::warn!("bf16 is not supported on the CPU backend; falling back to fp32");
    }
    let device = NdArrayDevice::default();
    info!("Using CPU backend");
    train::<Autodiff<NdArray<f32>>>(config, device)
}

#[cfg(not(feature = "backend-cpu"))]
fn run_train_cpu(_precision: PrecisionCli, _config: TrainingConfig) -> Result<()> {
    anyhow::bail!("cpu backend is not enabled. Rebuild with --features backend-cpu")
}

// ============================================================================
// Inference
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_infer(
    checkpoint: PathBuf,
    prompt: String,
    config: Option<PathBuf>,
    max_length: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    backend: Option<BackendCli>,
) -> Result<()> {
    let backend_choice = backend.unwrap_or(BackendCli::Wgpu);

    let params = SamplingParams::new()
        .with_temperature(temperature)
        .with_top_k(top_k)
        .with_top_p(top_p)
        .with_min_p(min_p)
        .with_repetition_penalty(repetition_penalty);

    match backend_choice {
        BackendCli::Wgpu => run_infer_wgpu(checkpoint, prompt, config, max_length, params),
        BackendCli::Cuda => run_infer_cuda(checkpoint, prompt, config, max_length, params),
        BackendCli::Cpu => run_infer_cpu(checkpoint, prompt, config, max_length, params),
    }
}

#[cfg(feature = "backend-wgpu")]
fn run_infer_wgpu(
    checkpoint: PathBuf,
    prompt: String,
    config: Option<PathBuf>,
    max_length: usize,
    params: SamplingParams,
) -> Result<()> {
    use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, Wgpu, WgpuDevice};

    let device = WgpuDevice::default();
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    info!("Using WGPU device: {:?}", device);

    run_infer_impl::<Wgpu<f32>>(checkpoint, prompt, config, max_length, params, device)
}

#[cfg(not(feature = "backend-wgpu"))]
fn run_infer_wgpu(
    _checkpoint: PathBuf,
    _prompt: String,
    _config: Option<PathBuf>,
    _max_length: usize,
    _params: SamplingParams,
) -> Result<()> {
    anyhow::bail!("wgpu backend is not enabled. Rebuild with --features backend-wgpu")
}

#[cfg(feature = "backend-cuda")]
fn run_infer_cuda(
    checkpoint: PathBuf,
    prompt: String,
    config: Option<PathBuf>,
    max_length: usize,
    params: SamplingParams,
) -> Result<()> {
    use burn::backend::cuda::{Cuda, CudaDevice};

    let device = CudaDevice::default();
    info!("Using CUDA device: {:?}", device);

    run_infer_impl::<Cuda<f32>>(checkpoint, prompt, config, max_length, params, device)
}

#[cfg(not(feature = "backend-cuda"))]
fn run_infer_cuda(
    _checkpoint: PathBuf,
    _prompt: String,
    _config: Option<PathBuf>,
    _max_length: usize,
    _params: SamplingParams,
) -> Result<()> {
    anyhow::bail!("cuda backend is not enabled. Rebuild with --features backend-cuda")
}

#[cfg(feature = "backend-cpu")]
fn run_infer_cpu(
    checkpoint: PathBuf,
    prompt: String,
    config: Option<PathBuf>,
    max_length: usize,
    params: SamplingParams,
) -> Result<()> {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    let device = NdArrayDevice::default();
    info!("Using CPU backend");

    run_infer_impl::<NdArray<f32>>(checkpoint, prompt, config, max_length, params, device)
}

#[cfg(not(feature = "backend-cpu"))]
fn run_infer_cpu(
    _checkpoint: PathBuf,
    _prompt: String,
    _config: Option<PathBuf>,
    _max_length: usize,
    _params: SamplingParams,
) -> Result<()> {
    anyhow::bail!("cpu backend is not enabled. Rebuild with --features backend-cpu")
}

fn run_infer_impl<B: burn::tensor::backend::Backend>(
    checkpoint: PathBuf,
    prompt: String,
    config: Option<PathBuf>,
    max_length: usize,
    params: SamplingParams,
    device: B::Device,
) -> Result<()> {
    // Load model config
    let model_config = load_model_config(&checkpoint, config.as_deref())?;
    info!(
        "Model config: hidden_size={}, n_layers={}, n_heads={}",
        model_config.hidden_size, model_config.n_layers, model_config.n_heads
    );

    // Load checkpoint
    let model = load_checkpoint::<B>(&checkpoint, &model_config, &device)?;

    // Generate
    info!("Prompt: {}", prompt);
    info!(
        "Sampling: temperature={}, top_k={}, top_p={}, min_p={}, rep_penalty={}",
        params.temperature, params.top_k, params.top_p, params.min_p, params.repetition_penalty
    );

    let generated = generate(&model, &prompt, max_length, &params, &device)?;

    println!("\n=== Generated Text ===");
    println!("{}{}", prompt, generated);
    println!("======================\n");

    Ok(())
}
