use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{activation::{silu, softmax}, backend::Backend, Int, Tensor},
};

use crate::config::ModelConfig;

/// RMSNorm implementation using modern Burn 0.19 APIs
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Scale parameter (gamma)
    weight: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability
    #[module(constant)]
    eps: f32,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        let weight = Param::from_tensor(Tensor::ones([dim], device));
        Self {
            weight,
            eps: 1e-6,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // Calculate RMS normalization
        // RMS = sqrt(mean(x^2) + eps)
        let x_squared = x.clone().powf_scalar(2.0);
        let mean_squared = x_squared.mean_dim(D - 1);
        let rms = (mean_squared + self.eps).sqrt();
        
        // Normalize and scale
        let x_normalized = x / rms;
        
        // Apply learned scale parameter
        x_normalized * self.weight.val().clone().unsqueeze()
    }
}

/// Modern RoPE implementation using efficient tensor operations
pub fn apply_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    position_ids: Tensor<B, 1, Int>,
    theta: f32,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let device = q.device();
    let [batch, seq_len, n_heads, head_dim] = q.dims();
    
    // Ensure head_dim is even for complex number representation
    assert!(head_dim % 2 == 0, "Head dimension must be even for RoPE");
    let half_dim = head_dim / 2;
    
    // Create frequency bands: theta^(-2i/d) for i in [0, d/2)
    let inv_freq = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect::<Vec<_>>();
    
    let inv_freq = Tensor::<B, 1>::from_floats(inv_freq, &device)
        .unsqueeze::<2>()  // [half_dim] -> [half_dim, 1]
        .repeat_dim(1, seq_len)  // [half_dim, seq_len]
        .transpose();  // [seq_len, half_dim]
    
    // Convert position IDs to float and compute angles
    let positions = position_ids.float().unsqueeze();  // [seq_len, 1]
    let freqs = positions.matmul(inv_freq.unsqueeze_dim::<3>(0));  // [seq_len, half_dim]
    
    // Compute sin and cos
    let cos = freqs.clone().cos();
    let sin = freqs.sin();
    
    // Expand for batch and heads dimensions
    let cos = cos
        .unsqueeze::<3>()  // [1, seq_len, half_dim]
        .unsqueeze::<4>()  // [1, seq_len, 1, half_dim]
        .repeat_dim(0, batch)
        .repeat_dim(2, n_heads);
    
    let sin = sin
        .unsqueeze::<3>()  // [1, seq_len, half_dim]
        .unsqueeze::<4>()  // [1, seq_len, 1, half_dim]
        .repeat_dim(0, batch)
        .repeat_dim(2, n_heads);
    
    // Apply rotation to q and k
    let q_rot = rotate_half(q, cos.clone(), sin.clone());
    let k_rot = rotate_half(k, cos, sin);
    
    (q_rot, k_rot)
}

/// Helper function to rotate embeddings
fn rotate_half<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, seq_len, n_heads, head_dim] = x.dims();
    let half = head_dim / 2;
    
    // Split x into two halves
    let x1 = x.clone().slice([0..batch, 0..seq_len, 0..n_heads, 0..half]);
    let x2 = x.slice([0..batch, 0..seq_len, 0..n_heads, half..head_dim]);
    
    // Rotate: [x1, x2] -> [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    let rotated_1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let rotated_2 = x1 * sin + x2 * cos;
    
    // Concatenate back
    Tensor::cat(vec![rotated_1, rotated_2], 3)
}

/// SwiGLU activation for FFN
#[derive(Module, Debug)]
pub struct SwiGlu<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> SwiGlu<B> {
    pub fn new(dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let gate_proj = LinearConfig::new(dim, hidden_dim)
            .with_bias(false)
            .init(device);
        let up_proj = LinearConfig::new(dim, hidden_dim)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(hidden_dim, dim)
            .with_bias(false)
            .init(device);
        
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Multi-head attention with RoPE
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    
    #[module(constant)]
    n_heads: usize,
    #[module(constant)]
    head_dim: usize,
    #[module(constant)]
    rope_theta: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        dim: usize,
        n_heads: usize,
        rope_theta: f32,
        device: &B::Device,
    ) -> Self {
        assert!(dim % n_heads == 0, "dim must be divisible by n_heads");
        let head_dim = dim / n_heads;
        
        let q_proj = LinearConfig::new(dim, dim)
            .with_bias(false)
            .init(device);
        let k_proj = LinearConfig::new(dim, dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(dim, dim)
            .with_bias(false)
            .init(device);
        let o_proj = LinearConfig::new(dim, dim)
            .with_bias(false)
            .init(device);
        
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            head_dim,
            rope_theta,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        position_ids: Tensor<B, 1, Int>,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();
        let device = x.device();
        
        // Project to Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);
        
        // Reshape to separate heads
        let q = q.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);
        let v = v.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);
        
        // Apply RoPE to Q and K
        let (q, k) = apply_rope(q, k, position_ids, self.rope_theta);
        
        // Transpose for attention: [batch, n_heads, seq_len, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);
        
        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(k.transpose()) / scale;
        
        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            scores + mask
        } else {
            scores
        };
        
        // Softmax
        let attn_weights = softmax(scores, 3);
        
        // Apply attention to values
        let output = attn_weights.matmul(v);
        
        // Transpose back and reshape
        let output = output
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);
        
        // Output projection
        self.o_proj.forward(output)
    }
}

/// Transformer block
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: Attention<B>,
    feed_forward: SwiGlu<B>,
    attention_norm: RmsNorm<B>,
    ffn_norm: RmsNorm<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let attention = Attention::new(
            config.hidden_size,
            config.n_heads,
            config.rope_theta,
            device,
        );
        
        let feed_forward = SwiGlu::new(
            config.hidden_size,
            config.intermediate_size,
            device,
        );
        
        let attention_norm = RmsNorm::new(config.hidden_size, device);
        let ffn_norm = RmsNorm::new(config.hidden_size, device);
        
        Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        position_ids: Tensor<B, 1, Int>,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        // Pre-norm attention with residual
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward(x, position_ids.clone(), mask);
        let x = residual + x;
        
        // Pre-norm FFN with residual
        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.feed_forward.forward(x);
        residual + x
    }
}

/// Llama model
#[derive(Module, Debug)]
pub struct LlamaModel<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
    
    #[module(constant)]
    config: ModelConfig,
}

impl<B: Backend> LlamaModel<B> {
    pub fn new(config: ModelConfig, device: &B::Device) -> Self {
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .init(device);
        
        let layers = (0..config.n_layers)
            .map(|_| TransformerBlock::new(&config, device))
            .collect();
        
        let norm = RmsNorm::new(config.hidden_size, device);
        
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);
        
        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
        }
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        position_ids: Option<Tensor<B, 1, Int>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();
        
        // Get position IDs if not provided
        let position_ids = position_ids.unwrap_or_else(|| {
            Tensor::arange(0..seq_len as i64, &device)
        });
        
        // Create causal mask
        let mask = create_causal_mask(seq_len, &device);
        
        // Token embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        
        // Apply transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, position_ids.clone(), Some(mask.clone()));
        }
        
        // Final norm and output projection
        hidden_states = self.norm.forward(hidden_states);
        self.lm_head.forward(hidden_states)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

/// Create causal attention mask
fn create_causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
    // Create a lower triangular matrix of ones
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = -10000.0; // Large negative value for masking
            }
        }
    }
    
    Tensor::from_data(mask_data, device)
        .reshape([seq_len, seq_len])
        .unsqueeze::<3>()  // Add batch dimension
        .unsqueeze::<4>()  // Add heads dimension
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_model_creation() {
        let device = Default::default();
        let config = ModelConfig::test();
        let model = LlamaModel::<TestBackend>::new(config, &device);
        
        // Test forward pass
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 10], &device);
        let output = model.forward(input_ids, None);
        
        assert_eq!(output.dims(), [2, 10, 256]); // batch, seq_len, vocab_size
    }
}
