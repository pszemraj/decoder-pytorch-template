//! Reference implementation for a Llama-style decoder.
//!
//! Use this file as a template when exploring new decoder ideas. Copy the
//! pieces you need into `src/models/my_model.rs`, adjust the architecture, and
//! export it through `models/mod.rs` so the trainer can pick it up.

use burn::{
    module::{Initializer, Module, Param},
    nn::{
        attention::generate_autoregressive_mask, Embedding, EmbeddingConfig, Linear, LinearConfig,
        RmsNorm, RmsNormConfig, RotaryEncoding, RotaryEncodingConfig,
    },
    tensor::{backend::Backend, Bool, DType, Int, Tensor},
};
use std::sync::Once;

use crate::{config::ModelConfig, tensor_utils::softmax_fp32_if_needed};

static BF16_LOG_ONCE: Once = Once::new();

fn mm_autocast_bf16<B: Backend>(
    x: Tensor<B, 3>,
    w_f32: &Param<Tensor<B, 2>>,
    use_bf16: bool,
) -> Tensor<B, 3> {
    // Expand weight for batched matmul: [hidden, out] -> [1, hidden, out]
    let w_f32_expanded = w_f32.val().unsqueeze_dim(0);
    if use_bf16 {
        let x_b = x.cast(DType::BF16);
        let w_b = w_f32_expanded.clone().cast(DType::BF16);
        BF16_LOG_ONCE
            .call_once(|| log::info!("BF16 GEMM active: x={:?} w={:?}", x_b.dtype(), w_b.dtype()));
        x_b.matmul(w_b).cast(DType::F32)
    } else {
        x.matmul(w_f32_expanded)
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    up_gate: Linear<B>,
    up_val: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let up_gate = LinearConfig::new(dim, hidden_dim)
            .with_bias(false)
            .init(device);
        let up_val = LinearConfig::new(dim, hidden_dim)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(hidden_dim, dim)
            .with_bias(false)
            .init(device);

        Self {
            up_gate,
            up_val,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, use_bf16_gemm: bool) -> Tensor<B, 3> {
        use burn::tensor::activation::silu;
        let a = mm_autocast_bf16(x.clone(), &self.up_gate.weight, use_bf16_gemm);
        let b = mm_autocast_bf16(x, &self.up_val.weight, use_bf16_gemm);
        let gated = silu(a) * b;
        mm_autocast_bf16(gated, &self.down_proj.weight, use_bf16_gemm)
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    rope: RotaryEncoding<B>,

    #[module(constant)]
    n_heads: usize,
    #[module(constant)]
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        assert!(
            config.hidden_size % config.n_heads == 0,
            "hidden_size must be divisible by n_heads"
        );
        let head_dim = config.hidden_size / config.n_heads;
        assert!(
            head_dim % 2 == 0,
            "attention head dimension must be even for RoPE"
        );

        let rope = RotaryEncodingConfig::new(config.max_position_embeddings, head_dim)
            .with_theta(config.rope_theta)
            .init(device);

        Self {
            q_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            rope,
            n_heads: config.n_heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        mask: &Tensor<B, 3, Bool>,
        position_offset: usize,
        use_bf16_gemm: bool,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();

        let q = mm_autocast_bf16(hidden_states.clone(), &self.q_proj.weight, use_bf16_gemm);
        let k = mm_autocast_bf16(hidden_states.clone(), &self.k_proj.weight, use_bf16_gemm);
        let v = mm_autocast_bf16(hidden_states, &self.v_proj.weight, use_bf16_gemm);

        let q = self.prepare(q, position_offset, batch_size, seq_len);
        let k = self.prepare(k, position_offset, batch_size, seq_len);
        let v = v
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);

        let mut scores = q.matmul(k.swap_dims(2, 3));
        let scale = (self.head_dim as f32).sqrt();
        scores = scores / scale;

        let causal_bias = mask
            .clone()
            .unsqueeze_dim(1)
            .float()
            .mul_scalar(-1e4)
            .cast(scores.dtype());
        let scores = scores + causal_bias;

        let attn_weights = softmax_fp32_if_needed(scores, 3);
        let context = attn_weights.matmul(v);

        let output =
            context
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);

        mm_autocast_bf16(output, &self.o_proj.weight, use_bf16_gemm)
    }

    fn prepare(
        &self,
        tensor: Tensor<B, 3>,
        position_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Tensor<B, 4> {
        let tensor = tensor
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2)
            .reshape([batch_size * self.n_heads, seq_len, self.head_dim]);
        let tensor = self.rope.apply(tensor, position_offset);
        tensor.reshape([batch_size, self.n_heads, seq_len, self.head_dim])
    }
}

/// Transformer block with RMSNorm + SwiGLU feed-forward.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: Attention<B>,
    feed_forward: FeedForward<B>,
    attention_norm: RmsNorm<B>,
    ffn_norm: RmsNorm<B>,
    #[module(constant)]
    use_bf16_gemm: bool,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &ModelConfig, device: &B::Device, use_bf16_gemm: bool) -> Self {
        let attention = Attention::new(config, device);
        let feed_forward =
            FeedForward::new(config.hidden_size, config.feedforward_hidden_size(), device);
        let attention_norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(1e-6)
            .init(device);
        let ffn_norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(1e-6)
            .init(device);

        Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            use_bf16_gemm,
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        mask: &Tensor<B, 3, Bool>,
        position_offset: usize,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.attention_norm.forward(hidden_states);
        let hidden_states =
            self.attention
                .forward(hidden_states, mask, position_offset, self.use_bf16_gemm);
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.ffn_norm.forward(hidden_states);
        let hidden_states = self.feed_forward.forward(hidden_states, self.use_bf16_gemm);
        residual + hidden_states
    }
}

#[derive(Module, Debug)]
pub struct LlamaModel<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    lm_head: Option<Linear<B>>,

    #[module(constant)]
    max_position_embeddings: usize,
    #[module(constant)]
    tie_embeddings: bool,
    #[module(constant)]
    use_bf16_gemm: bool,
}

impl<B: Backend> LlamaModel<B> {
    pub fn new(config: ModelConfig, device: &B::Device, use_bf16_gemm: bool) -> Self {
        let weight_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .with_initializer(weight_init.clone())
            .init(device);
        let layers = (0..config.n_layers)
            .map(|_| TransformerBlock::new(&config, device, use_bf16_gemm))
            .collect();
        let norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(1e-6)
            .init(device);
        let lm_head = if config.tie_embeddings {
            None
        } else {
            Some(
                LinearConfig::new(config.hidden_size, config.vocab_size)
                    .with_bias(false)
                    .with_initializer(weight_init)
                    .init(device),
            )
        };

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            max_position_embeddings: config.max_position_embeddings,
            tie_embeddings: config.tie_embeddings,
            use_bf16_gemm,
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>, position_offset: usize) -> Tensor<B, 3> {
        let [_batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();
        assert!(
            seq_len <= self.max_position_embeddings,
            "sequence length {} exceeds configured maximum {}",
            seq_len,
            self.max_position_embeddings
        );
        let position_offset =
            position_offset.min(self.max_position_embeddings.saturating_sub(seq_len));

        let mut hidden_states = self.embed_tokens.forward(input_ids);
        let mask = generate_autoregressive_mask::<B>(1, seq_len, &device);

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, &mask, position_offset);
        }

        hidden_states = self.norm.forward(hidden_states);
        self.project(hidden_states)
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn project(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.tie_embeddings {
            self.project_with_embeddings(hidden_states)
        } else {
            self.lm_head
                .as_ref()
                .expect("linear head should exist when embeddings aren't tied")
                .forward(hidden_states)
        }
    }

    fn project_with_embeddings(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden] = hidden_states.dims();
        let [vocab_size, _] = self.embed_tokens.weight.shape().dims();
        let flattened = hidden_states.reshape([batch_size * seq_len, hidden]);
        let weight = self.embed_tokens.weight.val().swap_dims(0, 1);
        flattened
            .matmul(weight)
            .reshape([batch_size, seq_len, vocab_size])
    }
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
        let model = LlamaModel::<TestBackend>::new(config, &device, false);

        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 10], &device);
        let output = model.forward(input_ids, 0);

        assert_eq!(output.dims(), [2, 10, 256]);
    }
}
