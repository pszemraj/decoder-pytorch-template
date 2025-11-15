use burn::config::Config;

/// Model configuration
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Size of vocabulary
    #[config(default = 256)]
    pub vocab_size: usize,

    /// Hidden size of the model
    #[config(default = 512)]
    pub hidden_size: usize,

    /// Number of hidden layers
    #[config(default = 8)]
    pub n_layers: usize,

    /// Number of attention heads
    #[config(default = 8)]
    pub n_heads: usize,

    /// Size of intermediate layer in FFN
    #[config(default = 2048)]
    pub intermediate_size: usize,

    /// RoPE theta parameter
    #[config(default = 10000.0)]
    pub rope_theta: f32,

    /// Maximum sequence length
    #[config(default = 2048)]
    pub max_position_embeddings: usize,

    /// Dropout probability
    #[config(default = 0.1)]
    pub dropout: f64,

    /// Tie token embedding weights to the LM head
    #[config(default = true)]
    pub tie_embeddings: bool,

    /// Optional multiplier for the feed-forward hidden size (takes precedence over `intermediate_size` when > 0)
    #[config(default = 0.0)]
    pub ffn_multiplier: f32,

    /// Round the feed-forward hidden size up to a multiple of this value.
    #[config(default = 256)]
    pub ffn_multiple_of: usize,
}

impl ModelConfig {
    /// Small model for testing
    pub fn test() -> Self {
        Self::new()
            .with_hidden_size(128)
            .with_n_layers(2)
            .with_n_heads(4)
            .with_intermediate_size(512)
    }

    /// Nano model (~10M params)
    pub fn nano() -> Self {
        Self::new()
            .with_hidden_size(384)
            .with_n_layers(6)
            .with_n_heads(6)
            .with_intermediate_size(576)
            .with_ffn_multiplier(1.5)
    }

    /// Small model (~100M params)
    pub fn small() -> Self {
        Self::new()
            .with_hidden_size(768)
            .with_n_layers(12)
            .with_n_heads(12)
            .with_intermediate_size(3072)
    }

    /// Base model (~350M params)
    pub fn base() -> Self {
        Self::new()
            .with_hidden_size(1024)
            .with_n_layers(24)
            .with_n_heads(16)
            .with_intermediate_size(4096)
    }

    /// Effective hidden size for the feed-forward block.
    pub fn feedforward_hidden_size(&self) -> usize {
        let base = if self.ffn_multiplier > 0.0 {
            (self.hidden_size as f32 * self.ffn_multiplier).round() as usize
        } else {
            self.intermediate_size.max(1)
        };
        let multiple = self.ffn_multiple_of.max(1);
        ((base + multiple - 1) / multiple) * multiple
    }
}

/// Training configuration
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Model configuration
    pub model: ModelConfig,

    /// Path to training data
    #[config(default = "String::from(\"data/train.txt\")")]
    pub train_data: String,

    /// Path to validation data
    #[config(default = "String::from(\"data/val.txt\")")]
    pub val_data: String,

    /// Batch size
    #[config(default = 8)]
    pub batch_size: usize,

    /// Sequence length
    #[config(default = 512)]
    pub sequence_length: usize,

    /// Number of optimizer steps
    #[config(default = 100000)]
    pub num_batches: usize,

    /// Learning rate
    #[config(default = 3e-4)]
    pub learning_rate: f64,

    /// Weight decay
    #[config(default = 0.01)]
    pub weight_decay: f32,

    /// Gradient clipping value
    #[config(default = 1.0)]
    pub gradient_clip: f32,

    /// Gradient accumulation steps
    #[config(default = 1)]
    pub gradient_accumulation_steps: usize,

    /// Validate every N steps
    #[config(default = 100)]
    pub validate_every: usize,

    /// Number of validation batches per evaluation
    #[config(default = 50)]
    pub val_batches: usize,

    /// Generation frequency (steps)
    #[config(default = 500)]
    pub generate_every: usize,

    /// Checkpoint frequency (steps)
    #[config(default = 5000)]
    pub save_every: usize,

    /// Length of each generated sample
    #[config(default = 256)]
    pub generation_length: usize,

    /// Number of prompt tokens to display before generation
    #[config(default = 128)]
    pub generation_prompt_length: usize,

    /// Random seed
    #[config(default = 42)]
    pub seed: u64,

    /// Output directory for checkpoints
    #[config(default = "String::from(\"checkpoints\")")]
    pub output_dir: String,

    /// Enable mixed precision training
    #[config(default = true)]
    pub mixed_precision: bool,
    /// Temperature for sampling
    #[config(default = 1.0)]
    pub temperature: f32,

    /// Min-p filtering threshold
    #[config(default = 0.1)]
    pub min_p: f32,

    /// Use gradient checkpointing to save memory
    #[config(default = false)]
    pub gradient_checkpointing: bool,
}

impl TrainingConfig {
    /// Configuration for quick testing
    pub fn test() -> Self {
        Self::new(ModelConfig::test())
            .with_batch_size(2)
            .with_sequence_length(128)
            .with_num_batches(1000)
            .with_learning_rate(1e-3)
    }

    /// Configuration for nano model
    pub fn nano() -> Self {
        Self::new(ModelConfig::nano())
            .with_batch_size(4)
            .with_sequence_length(512)
            .with_num_batches(10000)
            .with_learning_rate(3e-4)
            .with_gradient_accumulation_steps(4)
    }

    /// Configuration for small model
    pub fn small() -> Self {
        Self::new(ModelConfig::small())
            .with_batch_size(8)
            .with_sequence_length(1024)
            .with_num_batches(20000)
            .with_learning_rate(2e-4)
            .with_gradient_accumulation_steps(2)
            .with_gradient_checkpointing(true)
    }
}

/// Inference configuration
#[derive(Config, Debug)]
pub struct InferenceConfig {
    /// Model checkpoint path
    pub checkpoint_path: String,

    /// Maximum generation length
    #[config(default = 100)]
    pub max_length: usize,

    /// Temperature for sampling
    #[config(default = 0.8)]
    pub temperature: f32,

    /// Top-k filtering
    #[config(default = 50)]
    pub top_k: usize,

    /// Top-p (nucleus) filtering
    #[config(default = 0.9)]
    pub top_p: f32,

    /// Repetition penalty
    #[config(default = 1.2)]
    pub repetition_penalty: f32,

    /// Batch size for inference
    #[config(default = 1)]
    pub batch_size: usize,
}
