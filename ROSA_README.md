# ROSA Implementation Guide

This directory contains a complete implementation of **ROSA (Rapid Online Suffix Automaton)**, a neurosymbolic language model that replaces attention with deterministic recurrence pointers.

## Quick Start

### Training ROSA

```bash
# Train ROSA on enwik8 (character-level)
python train_rosa.py --config configs/rosa.yaml

# Quick test (create test config first)
python train_rosa.py --config configs/rosa_test.yaml
```

### Training Baseline (Llama for comparison)

```bash
# Train Llama baseline for comparison
python train.py --config configs/simple.yaml
```

## Architecture Overview

ROSA replaces the standard attention mechanism with:

1. **Online Suffix Automaton (OSA)**: Deterministic structure tracking all substrings
2. **Pointer Extraction**: Each token gets a discrete pointer to prior context
3. **Learned Mixing**: Neural envelope around symbolic core for trainability
4. **Feedforward**: Standard SwiGLU-style FFN layers

### Key Differences from Attention

| Aspect | Attention | ROSA |
|--------|-----------|------|
| Complexity | O(T²) | O(T) |
| Memory | KV cache per layer | Small pointer embeddings |
| Operation | Dense softmax | Deterministic links |
| Information | Soft, lossy | Lossless, exact |
| Interpretability | Low | High (explicit pointers) |

## Module Structure

```
decoder_pytorch/
├── rosa.py              # Main ROSA model
├── rosa_blocks.py       # ROSA residual blocks
├── rosa_automaton.py    # Online suffix automaton
└── rosa_relaxations.py  # Soft pointer mixing
```

## Configuration

ROSA-specific parameters in `configs/rosa.yaml`:

```yaml
# Core ROSA parameters
rosa_state_cap: 65536    # Max pointer states
k_candidates: 1          # Pointer candidates (>1 = soft mix)
temperature: 1.0         # Soft mixing temperature
channels: 1              # Parallel ROSA channels
ff_mult: 4               # FFN multiplier
drop: 0.1                # Dropout
```

### Parameter Tuning Guide

- **rosa_state_cap**: Maximum automaton states. Larger = more memory, better long context
- **k_candidates**:
  - `1`: Hard pointers (STE gradient)
  - `>1`: Soft pointer mixing during training
- **channels**:
  - `1`: Single recurrence stream
  - `>1`: Multi-channel (e.g., short/long span)
- **ff_mult**: Controls FFN size (4 = standard)

## API Compatibility

ROSA follows the same API as Llama:

```python
from decoder_pytorch import ROSA

# Initialize
model = ROSA(
    num_tokens=256,
    dim=512,
    depth=16,
    rosa_state_cap=65536,
)

# Forward pass
logits = model(input_ids)  # (B, T, V)
loss = model(input_ids, return_loss=True)

# Generation
output = model.generate(
    prompt,
    max_length=256,
    temperature=1.0,
    min_p=0.1,
)
```

## Performance Characteristics

### Expected Benefits

- **Linear time**: O(T) per layer vs O(T²) for attention
- **Long context**: Exact recurrence links across arbitrary spans
- **Interpretable**: Explicit pointer traces
- **Memory efficient**: No KV caches

### Current Limitations

- **Python OSA**: Pure Python automaton (slow). C++/CUDA kernels needed for production
- **Pointer optimality**: Simple suffix-link heuristic; could be improved
- **Empirical validation**: Needs extensive benchmarking vs attention baselines

## Training Tips

1. **Start simple**: Use `k_candidates=1` (hard pointers) initially
2. **Soft mixing**: Try `k_candidates=4` for better gradients during pretraining
3. **Multi-channel**: Experiment with `channels=2` for short/long span fusion
4. **Learning rate**: ROSA may benefit from slightly lower LR than attention
5. **Regularization**: Monitor pointer distributions; add entropy regularization if needed

## Evaluation Checklist

- [ ] Compare perplexity vs Llama at matched params/FLOPs
- [ ] Long-context retrieval (needle-in-haystack beyond train length)
- [ ] Analyze pointer distributions and recurrence patterns
- [ ] Measure tokens/sec throughput
- [ ] Ablate: k_candidates, temperature, channels, state_cap

## Implementation Notes

### Gradient Flow

- **No gradients** through OSA (discrete, symbolic)
- **Gradients flow** through:
  - Token embeddings
  - Pointer embeddings
  - Mixer projections (if k_candidates > 1)
  - Feedforward layers
  - LM head

### Straight-Through Estimator (STE)

When `k_candidates=1`, uses STE:
- Forward: Hard pointer index
- Backward: Identity gradient (as if continuous)

### Soft Mixing

When `k_candidates>1`:
- Forward: Learned softmax over K candidates
- Backward: Standard backprop through softmax
- Inference: Can harden with `hard=True` flag

## Future Optimizations

1. **CUDA kernel**: Implement `batch_pointers` in C++/CUDA
2. **Top-K recurrence**: Use K best recurrence paths instead of suffix link
3. **Adaptive pointers**: Learn when to use short vs long recurrence
4. **Hybrid**: Combine ROSA with sparse attention for best of both

## Citation

If you use ROSA, please cite the original concept document (update with actual paper when published):

```bibtex
@misc{rosa2025,
  title={ROSA: Rapid Online Suffix Automaton for Neurosymbolic Language Modeling},
  author={[Authors]},
  year={2025},
}
```

## References

- Online Suffix Automaton: Classic string indexing structure
- Straight-Through Estimator: Bengio et al. (2013)
- Min-p Sampling: https://arxiv.org/abs/2407.01082
