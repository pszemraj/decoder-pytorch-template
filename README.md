# Decoder PyTorch Template

A hackable template for autoregressive language model architecture experiments,  ~~batteries~~ Llama baseline included. Swap components, train and compare against new ideas.

Inspired by [Phil Wang](https://github.com/lucidrains)'s minimalist implementations.

## Quick Start

```bash
# get the code
git clone https://github.com/pszemraj/decoder-pytorch-template.git
cd decoder-pytorch-template
# activate your virtualenv (if not already)
pip install -e .
```

Train on the [included enwik8 dataset](data/README.md), character-level modeling:

```bash
# 100k batches on enwik8, 35M param Llama
python train.py --config configs/simple.yaml

# Nano run for CPU / MPS shakedowns (10k steps, L6 · H384 · ~9M params)
python train.py --config configs/nano.yaml

# Quick smoke test (tiny model, 10 batches)
python train.py --config configs/test.yaml
```

## Device Selection & Precision

- The training script calls `decoder_pytorch.get_optimal_device()` which prefers `cuda → mps → cpu`, returning `(device, device_type, amp_dtype)` and printing the accelerator picked.
- Override detection with `FORCE_DEVICE=cuda`, `FORCE_DEVICE=cpu`, or even `FORCE_DEVICE=cuda:1` to pick a specific index (also available as the `force=` argument).
- Mixed precision uses `torch.autocast` with `torch.bfloat16`; toggle via config if you want full fp32.

## Device Support

| Device        | Status | Notes                                               |
| ------------- | ------ | --------------------------------------------------- |
| NVIDIA GPU    | ✅      | Best performance, fused optimizer & flash attention |
| Apple Silicon | ✅      | Good performance, autocast can be flaky             |
| CPU           | ✅      | Slow but works; use `configs/nano.yaml`             |

## Structure

```text
decoder-pytorch-template/
├── decoder_pytorch/     # Model implementation
│   ├── llama.py        # Llama architecture
│   └── utils.py        # Sampling & device helpers
├── configs/            # Training configs
│   ├── simple.yaml     # Default config
│   ├── nano.yaml       # Quick CPU/MPS config
│   └── test.yaml       # Quick test config
├── data/
│   └── enwik8.gz       # Character-level dataset
└── train.py            # Training script
```

## Adding Your Architecture

To add your own model architecture:

1. **Create your model file**: Copy `decoder_pytorch/llama.py` to `decoder_pytorch/your_model.py`

2. **Implement required methods**: Your model class must have:
   - `__init__()` accepting at minimum: `num_tokens`, `dim`, `depth`, `heads`
   - `forward(x, mask=None, return_loss=False)` for training
   - `generate(prompt, max_length, temperature, filter_thres, min_p)` for inference
   - Properties: `vocab_size` and `model_dim`

3. **Export your model**: Update `decoder_pytorch/__init__.py`:

   ```python
   from .your_model import YourModel
   # Add to __all__ list
   ```

4. **Update training script**: Modify `train.py` line 16 and 88:

   ```python
   from decoder_pytorch import YourModel, model_summary
   # ...
   model = YourModel(
       num_tokens=config.get("num_tokens", 256),
       # ... other parameters
   )
   ```

5. **Configure and train**: Adjust `configs/simple.yaml` for your architecture's parameters

The included Llama baseline features:

- RMSNorm, SwiGLU, RoPE
- Proper weight initialization
- Generation with min-p sampling
- ~35M parameters (_default_)

## Configuration

Simple YAML [configs](configs/) control everything:

```yaml
# Model
dim: 512
depth: 16
heads: 8

# Training
num_batches: 100000
batch_size: 4
learning_rate: 0.003
```

## Design Philosophy

- **Simple** - No abstractions you don't need
- **Hackable** - Meant to be modified, not imported
- **Intuitive** - Focus on comparing/understanding architecture ideas instead of engineering details

## Requirements

> [!NOTE]
> bfloat16-compatible hardware[^1] is assumed in this codebase given its creation in 2025 AD.

[^1]: this means modern CPUs and ampere+ NVIDIA GPUs (compute capability ≥ 8.0)

Dependencies:

- PyTorch >= 2.9.0[^2]
- einops, pyyaml, tqdm
- [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch)

[^2]: If using PyTorch <2.9, you may need to adjust the bfloat16/autocast behaviour or fall back to full fp32 depending on hardware support.

## License

MIT

## Acknowledgments & References

Adapted from [lucidrains/nGPT-pytorch](https://github.com/lucidrains/nGPT-pytorch)

The Llama implementation is based on:

```bibtex
@misc{touvron2023llama2openfoundation,
      title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
      author={Hugo Touvron and Louis Martin and Kevin Stone and Peter Albert and Amjad Almahairi and Yasmine Babaei and Nikolay Bashlykov and Soumya Batra and Prajjwal Bhargava and Shruti Bhosale and Dan Bikel and Lukas Blecher and Cristian Canton Ferrer and Moya Chen and Guillem Cucurull and David Esiobu and Jude Fernandes and Jeremy Fu and Wenyin Fu and Brian Fuller and Cynthia Gao and Vedanuj Goswami and Naman Goyal and Anthony Hartshorn and Saghar Hosseini and Rui Hou and Hakan Inan and Marcin Kardas and Viktor Kerkez and Madian Khabsa and Isabel Kloumann and Artem Korenev and Punit Singh Koura and Marie-Anne Lachaux and Thibaut Lavril and Jenya Lee and Diana Liskovich and Yinghai Lu and Yuning Mao and Xavier Martinet and Todor Mihaylov and Pushkar Mishra and Igor Molybog and Yixin Nie and Andrew Poulton and Jeremy Reizenstein and Rashi Rungta and Kalyan Saladi and Alan Schelten and Ruan Silva and Eric Michael Smith and Ranjan Subramanian and Xiaoqing Ellen Tan and Binh Tang and Ross Taylor and Adina Williams and Jian Xiang Kuan and Puxin Xu and Zheng Yan and Iliyan Zarov and Yuchen Zhang and Angela Fan and Melanie Kambadur and Sharan Narang and Aurelien Rodriguez and Robert Stojnic and Sergey Edunov and Thomas Scialom},
      year={2023},
      eprint={2307.09288},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2307.09288},
}
```
