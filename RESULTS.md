# Benchmark Results: Backend & Precision Comparison

**Date:** 2026-01-18
**Model:** 8.95M parameter Llama-style decoder
**Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM)
**OS:** Ubuntu 24.04.3 LTS, kernel 6.14.0-37-generic
**Burn version:** 0.19.x (with CubeCL CUDA backend)

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
| WGPU    | fp32      | 360      | 2.78 it/s  | 7,625 MB  | 77%       | 5.51 → 1.54          | ✅ Good |
| CUDA    | fp32      | 136      | 7.35 it/s  | 9,034 MB  | 86%       | 5.52 → 1.58          | ✅ Good |
| CUDA    | bf16      | 49       | 20.4 it/s  | 5,815 MB  | 64%       | 5.51 → 3.49          | ⚠️ Convergence issue |

---

## Detailed Results

### 1. WGPU fp32 (Baseline)

**Training time:** 360 seconds (6:00)
**Throughput:** ~2.78 iterations/second
**Peak VRAM:** 7,625 MB
**Average GPU utilization:** 77%

**Validation loss progression:**
| Step | Val Loss |
|------|----------|
| 0    | 5.5119   |
| 200  | 2.4003   |
| 400  | 1.8745   |
| 600  | 1.7237   |
| 800  | 1.6141   |
| 1000 | 1.5445   |

**Sample generation (step 1000):**
```
Prompt: |.514
|-
||'''OBA'''||3<sup&
Generated: gt; |  255 || 37-2 || &#580; || 1993, | 256
```
Reasonable Wikipedia-style table continuation.

---

### 2. CUDA fp32

**Training time:** 136 seconds (2:16)
**Throughput:** ~7.35 iterations/second
**Peak VRAM:** 9,034 MB
**Average GPU utilization:** 86%

**Speedup vs WGPU:** 2.65× faster

**Validation loss progression:**
| Step | Val Loss |
|------|----------|
| 0    | 5.5202   |
| 200  | 2.3499   |
| 400  | 1.9310   |
| 600  | 1.7254   |
| 800  | 1.6433   |
| 1000 | 1.5772   |

**Sample generation (step 1000):**
```
Prompt: ommunity College]]
*[[North Iowa
Generated: nt alpsons to Salver Lecentury]]
*[[Pression combinocus]]
```
Reasonable Wikipedia link list continuation (some minor gibberish).

---

### 3. CUDA bf16 ⚠️

**Training time:** 49 seconds (0:49)
**Throughput:** ~20.4 iterations/second
**Peak VRAM:** 5,815 MB
**Average GPU utilization:** 64%

**Speedup vs CUDA fp32:** 2.78× faster
**VRAM reduction:** 36% less than CUDA fp32

**Validation loss progression:**
| Step | Val Loss |
|------|----------|
| 0    | 5.5141   |
| 200  | 3.5844   |
| 400  | 3.5273   |
| 600  | 3.4852   |
| 800  | 3.4977   |
| 1000 | 3.4938   |

**Sample generation (step 1000):**
```
Prompt: ]], film must be converted to 25
Generated:  air ta   nstt  esoorrrr]h hoto sem   e io mntidlrld e lr  tea o
```
**Gibberish output** - model failed to learn meaningful patterns.

### BF16 Convergence Issue Analysis

The bf16 run shows a critical convergence problem:
1. **Loss plateau:** Val loss drops to ~3.5 within 200 steps then stagnates
2. **No learning:** Loss barely improves from step 200 to 1000 (3.58 → 3.49)
3. **Generation quality:** Output is character-level noise, not coherent text
4. **Comparison:** FP32 variants reach 1.5-1.6 val loss with coherent generation

**Possible causes:**
- Gradient underflow in bf16 (7-bit mantissa vs 23-bit in fp32)
- Loss scaling not implemented in Burn's CubeCL backend
- RMSNorm or attention softmax precision issues
- Small model (8.95M params) more sensitive to precision loss

**Recommendation:** Use CUDA fp32 for training until bf16 precision issues are resolved in Burn/CubeCL. The 2.78× speedup is not worth the failed convergence.

---

## Performance Comparison

### Training Time
```
WGPU fp32:  ████████████████████████████████████ 360s
CUDA fp32:  █████████████▌                       136s (2.65× faster)
CUDA bf16:  ████▉                                 49s (7.35× faster than WGPU)
```

### VRAM Usage
```
WGPU fp32:  ████████████████████████             7,625 MB
CUDA fp32:  ████████████████████████████▌        9,034 MB (+18%)
CUDA bf16:  ██████████████████                   5,815 MB (-36% vs CUDA fp32)
```

### Model Quality (Final Val Loss)
```
WGPU fp32:  ████                                 1.54 ✅
CUDA fp32:  ████▏                                1.58 ✅
CUDA bf16:  ███████████                          3.49 ⚠️ (2.2× worse)
```

---

## Recommendations

1. **Production training:** Use **CUDA fp32** - 2.65× faster than WGPU with equivalent convergence
2. **Memory-constrained:** Still prefer CUDA fp32; bf16's memory savings aren't worth broken training
3. **Development/debugging:** WGPU fp32 works well and doesn't require CUDA feature flag
4. **BF16 status:** Not ready for production use with this model/framework combination

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

# CUDA bf16 (not recommended)
cargo run --release --features backend-cuda -- train configs/benchmark_1k.yaml --backend cuda --precision bf16
```
