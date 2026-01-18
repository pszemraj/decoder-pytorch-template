# Mixed Precision Status (Burn 0.20 + CubeCL 0.9)

BF16 training works correctly on CUDA. Here's how precision is handled:

## Matmul Accumulation

CubeCL 0.9 uses f32 accumulation for bf16 matmuls on non-macOS systems:

```rust
// From cubecl-matmul/src/components/spec.rs
impl MatmulPrecision for bf16 {
    type Lhs = (bf16, bf16);
    type Rhs = (bf16, bf16);
    #[cfg(not(target_os = "macos"))]
    type Acc = (bf16, f32);  // f32 accumulation
}
```

## Precision-Sensitive Operations

Operations that require higher precision are automatically upcasted:

| Operation | Implementation | Location |
|-----------|---------------|----------|
| RMSNorm | Burn upcasts to f32 internally | `burn-nn/src/modules/norm/rms.rs` |
| Softmax (attention) | `softmax_fp32_if_needed` helper | `src/tensor_utils.rs` |
| log_softmax (loss) | Upcasted in `cross_entropy` | `src/train.rs` |

## Backend Precision Support

| Backend | fp32 | bf16 |
|---------|------|------|
| CUDA | Yes | Yes (with f32 accum) |
| WGPU | Yes | No |
| CPU | Yes | No |

## Benchmark Results

See `RESULTS.md` for detailed benchmarks. Summary on RTX 5090:

- CUDA fp32: 87s, val loss 1.58
- CUDA bf16: 108s, val loss 1.61 (equivalent quality)
- WGPU fp32: 280s, val loss 1.59

BF16 is slightly slower than fp32 for this small model due to fp32 upcasting overhead in loss computation. For larger models, GEMM speedup should dominate.
