use anyhow::Result;
use burn::backend::Autodiff;
use burn_llama::{train, TrainingConfig};
use clap::{Parser, ValueEnum};
use half::bf16;
#[cfg(feature = "backend-cpu")]
use log::warn;
use log::{info, LevelFilter};

/// Minimal launcher: `cargo run --release -- configs/test.yaml`
#[derive(Parser, Debug)]
#[command(name = "burn-llama")]
#[command(about = "Train decoder experiments from a YAML config", long_about = None)]
struct Cli {
    /// Path to YAML training config (defaults to configs/test.yaml)
    #[arg(value_name = "CONFIG", default_value = "configs/test.yaml")]
    config: String,

    /// Backend to use (wgpu, cuda, cpu)
    #[arg(long, value_enum)]
    backend: Option<BackendCli>,

    /// Numeric precision (fp32, bf16)
    #[arg(long, value_enum)]
    precision: Option<PrecisionCli>,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
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
        .filter_level(LevelFilter::Warn)
        .filter_module("burn_llama", log_level)
        .init();

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

    let backend_choice = cli.backend.unwrap_or(BackendCli::Wgpu);
    let precision_choice = cli.precision.unwrap_or(if training_config.mixed_precision {
        PrecisionCli::Bf16
    } else {
        PrecisionCli::Fp32
    });

    info!("Starting training with config: {:#?}", training_config);
    run_backend(backend_choice, precision_choice, training_config)?;

    Ok(())
}

fn load_config(path: &str) -> Result<TrainingConfig> {
    let config_str = std::fs::read_to_string(path)?;
    let config: TrainingConfig = serde_yaml::from_str(&config_str)?;
    Ok(config)
}

fn run_backend(backend: BackendCli, precision: PrecisionCli, config: TrainingConfig) -> Result<()> {
    match backend {
        BackendCli::Wgpu => run_wgpu(precision, config),
        BackendCli::Cuda => run_cuda(precision, config),
        BackendCli::Cpu => run_cpu(precision, config),
    }
}

#[cfg(feature = "backend-wgpu")]
fn run_wgpu(precision: PrecisionCli, config: TrainingConfig) -> Result<()> {
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
fn run_wgpu(_precision: PrecisionCli, _config: TrainingConfig) -> Result<()> {
    anyhow::bail!("wgpu backend is not enabled. Rebuild with --features backend-wgpu")
}

#[cfg(feature = "backend-cuda")]
fn run_cuda(precision: PrecisionCli, config: TrainingConfig) -> Result<()> {
    use burn::backend::cuda::{Cuda, CudaDevice};

    let device = CudaDevice::default();
    info!("Using CUDA device: {:?}", device);

    match precision {
        PrecisionCli::Fp32 => train::<Autodiff<Cuda<f32>>>(config, device),
        PrecisionCli::Bf16 => train::<Autodiff<Cuda<bf16>>>(config, device),
    }
}

#[cfg(not(feature = "backend-cuda"))]
fn run_cuda(_precision: PrecisionCli, _config: TrainingConfig) -> Result<()> {
    anyhow::bail!("cuda backend is not enabled. Rebuild with --features backend-cuda")
}

#[cfg(feature = "backend-cpu")]
fn run_cpu(precision: PrecisionCli, config: TrainingConfig) -> Result<()> {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    if matches!(precision, PrecisionCli::Bf16) {
        warn!("bf16 is not supported on the CPU backend; falling back to fp32");
    }
    let device = NdArrayDevice::default();
    info!("Using CPU backend");
    train::<Autodiff<NdArray<f32>>>(config, device)
}

#[cfg(not(feature = "backend-cpu"))]
fn run_cpu(_precision: PrecisionCli, _config: TrainingConfig) -> Result<()> {
    anyhow::bail!("cpu backend is not enabled. Rebuild with --features backend-cpu")
}
