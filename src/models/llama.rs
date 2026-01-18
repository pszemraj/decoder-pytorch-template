//! Reference implementation for a Llama-style decoder.
//!
//! Use this file as a template when exploring new decoder ideas. Copy the
//! pieces you need into `src/models/my_model.rs`, adjust the architecture, and
//! export it through `models/mod.rs` so the trainer can pick it up.

use burn::{
    module::{AutodiffModule, Initializer, Module, ModuleDisplay, ModuleDisplayDefault, Param},
    nn::{
        attention::generate_autoregressive_mask, Embedding, EmbeddingConfig, Linear, LinearConfig,
        RmsNorm, RmsNormConfig, RotaryEncoding, RotaryEncodingConfig,
    },
    record::{PrecisionSettings, Record},
    tensor::{backend::Backend, Bool, DType, Int, Tensor},
};
use serde::{Deserialize, Serialize};
use std::sync::Once;

use crate::{config::ModelConfig, tensor_utils::softmax_fp32_if_needed};

static GEMM_LOG_ONCE: Once = Once::new();

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum GemmMode {
    Fp32,
    Flex32,
    Bf16,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MpPolicy {
    pub qkv: GemmMode,
    pub o: GemmMode,
    pub ffn_up: GemmMode,
    pub ffn_down: GemmMode,
}

impl<B: Backend> Module<B> for MpPolicy {
    type Record = MpPolicy;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        self
    }
}

impl<B: Backend> Record<B> for MpPolicy {
    type Item<S: PrecisionSettings> = MpPolicy;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}

impl<B: burn::tensor::backend::AutodiffBackend> AutodiffModule<B> for MpPolicy {
    type InnerModule = MpPolicy;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl ModuleDisplayDefault for MpPolicy {
    fn content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
        Some(
            content
                .add("qkv", &format!("{:?}", self.qkv))
                .add("o", &format!("{:?}", self.o))
                .add("ffn_up", &format!("{:?}", self.ffn_up))
                .add("ffn_down", &format!("{:?}", self.ffn_down)),
        )
    }
}

impl ModuleDisplay for MpPolicy {}

impl MpPolicy {
    pub fn fp32() -> Self {
        Self {
            qkv: GemmMode::Fp32,
            o: GemmMode::Fp32,
            ffn_up: GemmMode::Fp32,
            ffn_down: GemmMode::Fp32,
        }
    }

    pub fn bf16() -> Self {
        Self {
            qkv: GemmMode::Bf16,
            o: GemmMode::Bf16,
            ffn_up: GemmMode::Bf16,
            ffn_down: GemmMode::Bf16,
        }
    }

    /// Stable mixed default for CUDA today: attention in Flex32 (TF32 compute, F32 accum),
    /// FFN kept in fp32 unless explicitly relaxed.
    pub fn attn_flex32_ffn_fp32() -> Self {
        Self {
            qkv: GemmMode::Flex32,
            o: GemmMode::Flex32,
            ffn_up: GemmMode::Fp32,
            ffn_down: GemmMode::Fp32,
        }
    }

    /// More aggressive: keep attention Flex32, allow BF16 on FFN up, Flex32 on down.
    pub fn attn_flex32_ffn_mix() -> Self {
        Self {
            qkv: GemmMode::Flex32,
            o: GemmMode::Flex32,
            ffn_up: GemmMode::Bf16,
            ffn_down: GemmMode::Flex32,
        }
    }
}

fn linear_gemm_autocast<B: Backend>(
    x: Tensor<B, 3>,
    w_f32: &Param<Tensor<B, 2>>,
    mode: GemmMode,
) -> Tensor<B, 3> {
    let [b, s, in_d] = x.dims();
    let w = w_f32.val();
    let [in_w, out_d] = w.dims();
    debug_assert_eq!(
        in_d, in_w,
        "in dim mismatch: x last={} vs w first={}",
        in_d, in_w
    );

    // Flatten to 2D to force the tuned GEMM path.
    let x2 = x.reshape([b * s, in_d]);
    let y2 = match mode {
        GemmMode::Fp32 => x2.matmul(w),
        GemmMode::Flex32 => {
            let x_c = x2.cast(DType::Flex32);
            let w_c = w.clone().cast(DType::Flex32);
            GEMM_LOG_ONCE.call_once(|| log::warn!("GEMM mode = Flex32 (TF32 compute, F32 accum)"));
            x_c.matmul(w_c).cast(DType::F32)
        }
        GemmMode::Bf16 => {
            let x_b = x2.cast(DType::BF16);
            let w_b = w.clone().cast(DType::BF16);
            GEMM_LOG_ONCE
                .call_once(|| log::warn!("GEMM mode = BF16 (BF16 compute, F32 accum)"));
            x_b.matmul(w_b)
        }
    };
    y2.reshape([b, s, out_d])
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

    pub fn forward(&self, x: Tensor<B, 3>, mp_policy: MpPolicy) -> Tensor<B, 3> {
        use burn::tensor::activation::silu;
        let a = linear_gemm_autocast(x.clone(), &self.up_gate.weight, mp_policy.ffn_up);
        let b = linear_gemm_autocast(x, &self.up_val.weight, mp_policy.ffn_up);
        let gated = silu(a) * b;
        linear_gemm_autocast(gated, &self.down_proj.weight, mp_policy.ffn_down)
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
        mp_policy: MpPolicy,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();

        let q = linear_gemm_autocast(hidden_states.clone(), &self.q_proj.weight, mp_policy.qkv);
        let k = linear_gemm_autocast(hidden_states.clone(), &self.k_proj.weight, mp_policy.qkv);
        let v = linear_gemm_autocast(hidden_states, &self.v_proj.weight, mp_policy.qkv);

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

        linear_gemm_autocast(output, &self.o_proj.weight, mp_policy.o)
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
    mp_policy: MpPolicy,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &ModelConfig, device: &B::Device, mp_policy: MpPolicy) -> Self {
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
            mp_policy,
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
                .forward(hidden_states, mask, position_offset, self.mp_policy);
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.ffn_norm.forward(hidden_states);
        let hidden_states = self.feed_forward.forward(hidden_states, self.mp_policy);
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
    mp_policy: MpPolicy,
}

impl<B: Backend> LlamaModel<B> {
    pub fn new(config: ModelConfig, device: &B::Device, mp_policy: MpPolicy) -> Self {
        let weight_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .with_initializer(weight_init.clone())
            .init(device);
        let layers = (0..config.n_layers)
            .map(|_| TransformerBlock::new(&config, device, mp_policy))
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
            mp_policy,
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
        let model = LlamaModel::<TestBackend>::new(config, &device, MpPolicy::fp32());

        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 10], &device);
        let output = model.forward(input_ids, 0);

        assert_eq!(output.dims(), [2, 10, 256]);
    }
}
