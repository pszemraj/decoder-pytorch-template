# Benchmark Results: Backend & Precision Comparison

**Date:** 2026-01-18
**Model:** 8.95M parameter Llama-style decoder
**Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM)
**OS:** Ubuntu 24.04.3 LTS, kernel 6.14.0-37-generic
**Burn version:** 0.20.0 (with CubeCL 0.9.0)

## Test Configuration

```yaml
model:
  hidden_size: 384
  n_layers: 6
  n_heads: 6
  vocab_size: 256

batch_size: 4
sequence_length: 512
gradient_accumulation_steps: 4
num_batches: 1000
learning_rate: 0.0003
seed: 42
```

**Effective batch size:** 4 × 4 = 16 sequences per optimizer step
**Total tokens trained:** ~8.2M tokens

---

## Summary Table

| Backend | Precision | Time (s) | Throughput | Peak VRAM | Avg GPU % | Val Loss (init→final) | Status |
|---------|-----------|----------|------------|-----------|-----------|----------------------|--------|
| WGPU    | fp32      | 250.10   | 4.00 it/s  | n/a       | n/a       | 5.5447 → 1.6549       | ✅ Good |
| CUDA    | fp32      | 85.35    | 11.71 it/s | n/a       | n/a       | 5.5404 → 1.5443       | ✅ Good |
| CUDA    | bf16      | 106.07   | 9.43 it/s  | n/a       | n/a       | 5.5547 → 1.5688       | ✅ Good |

---

## Detailed Results

### 1. WGPU fp32 (Baseline)

**Training time:** 250.10 seconds (4:10)
**Throughput:** ~4.00 iterations/second
**Peak VRAM:** n/a (not measured)
**Average GPU utilization:** n/a (not measured)

**Validation loss progression:**
| Step | Val Loss |
|------|----------|
| 0    | 5.5447   |
| 200  | 2.3589   |
| 400  | 1.9336   |
| 600  | 1.7167   |
| 800  | 1.5961   |
| 1000 | 1.6549   |

**Sample generation (step 1000):**
```
Prompt: revision>
  </page>
  <page>
   
Generated:  <title>Gethol</title>
    <id>16502</id>
    <revision>
      <
```

---

### 2. CUDA fp32

**Training time:** 85.35 seconds (1:25)
**Throughput:** ~11.71 iterations/second
**Peak VRAM:** n/a (not measured)
**Average GPU utilization:** n/a (not measured)

**Speedup vs WGPU:** 2.93× faster

**Validation loss progression:**
| Step | Val Loss |
|------|----------|
| 0    | 5.5404   |
| 200  | 2.3722   |
| 400  | 1.8792   |
| 600  | 1.7168   |
| 800  | 1.5733   |
| 1000 | 1.5443   |

**Sample generation (step 1000):**
```
Prompt: |2&lt;sup&gt;ND&lt;/sup&gt;||46

Generated: |--
|-----------------------------------------------------------
```

---

### 3. CUDA bf16 ✅

**Training time:** 106.07 seconds (1:46)
**Throughput:** ~9.43 iterations/second
**Peak VRAM:** n/a (not measured)
**Average GPU utilization:** n/a (not measured)

**Comparison to CUDA fp32:**
- 24% slower (106s vs 85s) - overhead from fp32 upcasting in loss computation
- Equivalent convergence (val loss ~1.57)

**Validation loss progression:**
| Step | Val Loss |
|------|----------|
| 0    | 5.5547   |
| 200  | 2.4719   |
| 400  | 1.9859   |
| 600  | 1.7941   |
| 800  | 1.6551   |
| 1000 | 1.5688   |

**Sample generation (step 1000):**
```
Prompt: y has also recovered somewhat si
Generated: nce several completing a star the construction of consideral ear
```

### BF16 Implementation Notes

BF16 training now works correctly with Burn 0.20. Key implementation details:

1. **CubeCL matmul uses f32 accumulation** - Defined in `cubecl-matmul/src/components/spec.rs`:
   ```rust
   impl MatmulPrecision for bf16 {
       type Acc = (bf16, f32);  // f32 accumulation on non-macOS
   }
   ```

2. **RMSNorm upcasts to fp32** - Burn's RmsNorm automatically computes in fp32 for numerical stability

3. **Softmax attention upcasts to fp32** - Our `softmax_fp32_if_needed` helper handles this

4. **Loss computation upcasts to fp32** - Cross-entropy uses fp32 for log_softmax to prevent underflow:
   ```rust
   let logits_compute = if use_fp32 {
       logits.cast(DType::F32)
   } else { logits };
   let log_probs = log_softmax(logits_compute, 1);
   ```

The fp32 upcasting in the loss adds overhead, making bf16 slightly slower than fp32 for this small model. For larger models, the GEMM speedup should dominate and bf16 would be faster.

---

## Performance Comparison

### Training Time
```
WGPU fp32:  ████████████████████████████████████████████████ 250s
CUDA bf16:  ██████████████████▌                              106s (2.36× faster)
CUDA fp32:  ██████████████▊                                   85s (2.93× faster)
```

### VRAM Usage
Not measured in this run.

### Model Quality (Final Val Loss)
```
CUDA fp32:  ████                                             1.54 ✅
WGPU fp32:  ████                                             1.65 ✅
CUDA bf16:  ████                                             1.57 ✅
```

All configurations produce similar final loss, demonstrating equivalent training quality.

---

## Recommendations

1. **Production training (CUDA available):** Use **CUDA fp32** - 2.93× faster than WGPU
2. **Memory-constrained (CUDA):** Use **CUDA bf16** - similar quality, potential VRAM savings (measure on your setup)
3. **Portable/development:** Use **WGPU fp32** - works without CUDA feature flag
4. **Larger models:** BF16 should be faster due to GEMM speedup dominating the overhead

---

## Version History

### Burn 0.20 (Current)
- BF16 works correctly with proper convergence
- CUDA ~35% faster than Burn 0.19
- CubeCL 0.9.0 with improved memory management

### Burn 0.19 (Previous)
- BF16 had convergence issues (loss stuck at ~3.5)
- BufferTooBig errors with bf16 fusion
- CUDA fp32: 136s, WGPU fp32: 360s

---

## Environment Details

```
GPU: NVIDIA GeForce RTX 5090 (32,607 MB total VRAM)
Driver: 575.64.03
CUDA: Via Burn CubeCL backend
CPU: AMD Ryzen 7 7700X (8-core)
RAM: 64 GB
```

## Reproduction

```bash
# WGPU fp32
cargo run --release -- train configs/benchmark_1k.yaml --backend wgpu --precision fp32

# CUDA fp32
cargo run --release --features backend-cuda -- train configs/benchmark_1k.yaml --backend cuda --precision fp32

# CUDA bf16
cargo run --release --features backend-cuda -- train configs/benchmark_1k.yaml --backend cuda --precision bf16
```
