# Decoder Burn Template

Rust re-implementation of the PyTorch decoder playground from the `main` branch: Llama baseline included, easy to hack and compare new ideas, but running on Burn 0.19 backends (WGPU, CUDA, or CPU).

## Highlights

- **One binary, YAML configs** - point at `configs/*.yaml` and train.
- **Backend-agnostic** - select WGPU, CUDA, or CPU at runtime (with matching Cargo feature).
- **Precision toggle** - run in fp32 or bf16 (autodetected from config, overridable via CLI).
- **Llama-style decoder** - RMSNorm, SwiGLU, RoPE, causal mask, optional tied embeddings.
- **Training parity with PyTorch** - gradient accumulation matches the Python version token-for-token.
- **Auto dataset streaming** - includes `data/enwik8.gz`; no preprocessing required.

> ⚠️ Mixed precision (CUDA): FP32 is the supported path. BF16/TF32 attention kernels in this build do **not** reach FP32 convergence. `--precision bf16` will run attention/FFN in FP32 by default and emit a warning. Experimental overrides: `ATTN_MODE=flex32` (TF32 compute, FP32 accum) or `ATTN_MODE=bf16` (known to diverge). Expect degraded loss/generations with overrides until Burn exposes BF16-with-FP32-accum GEMMs.

> ℹ️ Flex32 note: Burn’s `Flex32` is *not* NVIDIA TF32. It stores values with F16 mantissa/range and computes in F32; attention still loses accuracy. Stick to FP32 for convergence.

## Quick Start

```bash
# Clone
git clone https://github.com/pszemraj/decoder-pytorch-template.git
cd decoder-pytorch-template

# Smoke test (WGPU backend, fp32)
cargo run --release -- configs/test.yaml

# Nano config on CUDA in bf16 (requires --features backend-cuda)
cargo run --release --features backend-cuda -- \
    --backend cuda --precision bf16 configs/nano.yaml

# CPU baseline (requires --features backend-cpu)
cargo run --release --features backend-cpu -- \
    --backend cpu --precision fp32 configs/test.yaml

# Custom config
cargo run --release -- path/to/my_config.yaml
```

Both sample configs stream `data/enwik8.gz`, randomly slicing fixed-length sequences. Use `train_steps_per_epoch` / `val_steps` to cap iterations for quick experiments.

### CLI Flags

- `--backend {wgpu|cuda|cpu}`: pick the backend. `wgpu` is default. Remember to enable the matching Cargo feature (`backend-cuda`, `backend-cpu`).
- `--precision {fp32|bf16}`: overrides numeric precision. If omitted, the YAML's `mixed_precision` flag selects `bf16` when true, `fp32` otherwise. CPU always falls back to `fp32`.

## Configuration Files

Every entry in `configs/*.yaml` feeds directly into `TrainingConfig` and `ModelConfig`. Example:

```yaml
model:
  vocab_size: 256
  hidden_size: 768
  n_layers: 12
  n_heads: 12
  intermediate_size: 3072
  ffn_multiplier: 4.0  # optional override for FFN size (hidden_size * multiplier)
  rope_theta: 10000.0
  max_position_embeddings: 2048
  tie_embeddings: true  # share token embed and LM head weights

train_data: data/enwik8.gz  # .gz automatically split 90/10
val_data: data/enwik8.gz

batch_size: 8
sequence_length: 1024
num_epochs: 20
learning_rate: 2e-4
weight_decay: 0.01
gradient_clip: 1.0
gradient_accumulation_steps: 4
train_steps_per_epoch: 2000  # 0 = iterate entire dataset
val_steps: 200
mixed_precision: true
output_dir: runs/my-exp
```

## Project Layout

```
decoder-burn-template/
├── src/
│   ├── models/
│   │   ├── llama.rs   # Reference decoder (RoPE, SwiGLU, RMSNorm)
│   │   └── mod.rs     # Re-export point for your custom models
│   ├── train.rs       # Training loop, dataset loaders, sampling
│   ├── data.rs        # Char dataset + gzip loader
│   ├── config.rs      # Burn Config structs
│   └── main.rs        # CLI entry (backend/precision switches)
├── configs/           # YAML experiments (test, nano, …)
├── data/enwik8.gz     # Sample dataset (character-level)
└── runs/              # Logs + checkpoints (final.bin per run)
```

### Adding Your Model

1. Copy `src/models/llama.rs` `src/models/my_model.rs`.
2. Adjust the architecture (attention, FFN, etc.) and expose it via `src/models/mod.rs`.
3. In `train.rs`, swap the `LlamaModel` type alias (or add a CLI flag if you want runtime switching).
4. Update your YAML (`model.*` field names are forwarded to the new config).

## Device & Precision Matrix

| Backend Flag   | Runtime Flag     | Precision options | Notes                                  |
| -------------- | ---------------- | ----------------- | -------------------------------------- |
| `backend-wgpu` | `--backend wgpu` | fp32 / bf16       | Default build, Vulkan/WebGPU/Metal     |
| `backend-cuda` | `--backend cuda` | fp32 / bf16       | Requires NVIDIA GPU + CUDA libs        |
| `backend-cpu`  | `--backend cpu`  | fp32 (bf16fp32)   | Uses NdArray backend (slow but simple) |

## Feature Parity with `train.py`

- **Gradient accumulation**: Burn implementation sums per-token loss over each micro-batch (CrossEntropy returns the mean), scales by `1/grad_accum`, and only steps after `grad_accum_every` iterations-mathematically identical to PyTorch's "sum, divide once" approach. Token counts are fixed (`batch_size * seq_len`), so no scaling drift.
- **Tied embeddings**: Configurable via `model.tie_embeddings`. When enabled, the LM head reuses the embedding weight, matching the PyTorch baseline.
- **RoPE/SwiGLU/RMSNorm**: Same layout and initialization (Normal(0,0.02) for embeddings/head) as the original codebase.
- **Automatic dataset handling**: `.gz` files are streamed and split 90/10, just like `load_data` in `train.py`.
- **Checkpointing**: Each epoch logs `Step X | Epoch Y | Val loss`, saves best-in-run checkpoints, and writes `runs/<name>/final.bin` via `CompactRecorder`.

## Tips & Troubleshooting

- **Slow hardware / limited memory**: lower `batch_size`, raise `gradient_accumulation_steps`, or reduce `sequence_length`. Use `train_steps_per_epoch` to keep epochs short.
- **Precision mismatch**: if gradients blow up in bf16, switch the YAML's `mixed_precision` to `false` or run with `--precision fp32`.
- **Backend missing**: recompile with the appropriate `--features backend-*` flag; WGPU is the only backend enabled by default.
- **Custom datasets**: point `train_data` / `val_data` at your files (plain text or gzip). The `CharDataset` takes care of random slicing and padding.

## License & Credits

MIT, same as the original project. Heavily inspired by the PyTorch template in `main`, ported to Burn to explore novel decoder architectures with a different toolchain.
4. Use smaller model preset

### Slow Training

1. Ensure release mode: `cargo run --release`
2. Check GPU utilization
3. Increase batch size if memory allows
4. Enable fusion backend

### Compilation Errors

Ensure you have Burn 0.19:

```toml
burn = { version = "0.19", features = ["std", "train", "wgpu"] }
```

- If you bump into an `evaluate_obligation` incremental ICE from `rustc` (seen as "encountered incremental compilation error ... canonical"), it's a compiler bug. Incremental builds are now disabled for dev/test profiles in `Cargo.toml`; if you manually re-enable them, run `cargo clean -p burn-llama` to unstick the build or set `CARGO_INCREMENTAL=0`.

## Roadmap

- [x] Basic Llama architecture
- [x] Efficient RoPE implementation
- [x] Gradient accumulation
- [x] Mixed precision training
- [ ] Flash Attention
- [ ] Streaming generation

Maybe:

- [ ] Distributed training
- [ ] ONNX export
- [ ] Quantization (INT8/INT4)
- [ ] KV cache for inference

## Contributing

Contributions are welcome! Please ensure:

1. Code follows Rust idioms
2. Tests pass: `cargo test`
3. Format: `cargo fmt`
4. Lint: `cargo clippy`

## License

MIT

## Acknowledgments

- Burn framework: <https://burn.dev>
- Original Llama paper: <https://arxiv.org/abs/2302.13971>
- PyTorch template inspiration: <https://github.com/pszemraj/decoder-pytorch-template>

## Citation

```bibtex
@software{szemraj2025decoderburn,
  title = {Burn Llama: Modern Implementation with Burn 0.19},
  author = {Peter Szemraj},
  year = {2025},
  url = {https://github.com/pszemraj/decoder-burn-template}
}
```
