# Mixed Precision Status (CUDA)

Summary of why `--precision bf16` is unsafe today and how Flex32 differs from NVIDIA TF32.

- FP32 training converges as expected. Use this by default.
- `Flex32` in Burn is **not** NVIDIA TF32. It stores values with F16 mantissa/range and computes in F32 (see `burn-tensor` `flex32` element), so attention still loses accuracy.
- BF16 matmuls currently accumulate in BF16. There is no BF16-input + FP32-accum kernel exposed in CubeCL matmul yet, so attention diverges.
- Because of the above, the runner forces FP32 attention/FFN even when `--precision bf16` is set. Overrides:
  - `ATTN_MODE=flex32` to force Flex32/TF32 compute in attention (still lower accuracy than FP32).
  - `ATTN_MODE=bf16` to force raw BF16 GEMMs (known to diverge; for experiments only).
- A proper solution would add FP32 accumulation control to CubeCL matmuls (analogous to how reductions pick FP32 accum for BF16/F16). Until then, treat CUDA “bf16” as experimental only.

If/when CubeCL gains BF16-input + FP32-accum matmuls, this project can switch back to real mixed precision for speed without breaking convergence.
