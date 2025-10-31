import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# --------------------------------------------------------------------------
# Sampling utilities
# --------------------------------------------------------------------------


def log(t: Tensor, eps: float = 1e-20) -> Tensor:
    """Safe log operation."""
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t: Tensor) -> Tensor:
    """Generate Gumbel noise."""
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    t: Tensor, temperature: float = 1.0, dim: int = -1, keepdim: bool = True
) -> Tensor:
    """Sample using Gumbel-max trick.

    Args:
        t: Logits tensor
        temperature: Sampling temperature
        dim: Dimension to sample from
        keepdim: Keep dimension after argmax

    Returns:
        Sampled indices
    """
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(
        dim=dim, keepdim=keepdim
    )


def min_p_filter(logits: Tensor, min_p: float = 0.1) -> Tensor:
    """Apply min-p filtering to logits.

    Min-p filtering masks tokens with probability less than min_p * max_prob.
    Paper: https://arxiv.org/abs/2407.01082

    Args:
        logits: Logits tensor
        min_p: Minimum probability threshold (relative to max)

    Returns:
        Filtered logits
    """
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float("-inf"), logits)


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Apply top-k filtering to logits.

    Args:
        logits: Logits tensor
        k: Number of top tokens to keep

    Returns:
        Filtered logits
    """
    if k <= 0:
        return logits

    values, _ = torch.topk(logits, k, dim=-1)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, float("-inf"), logits)


def top_p_filter(logits: Tensor, p: float = 0.9) -> Tensor:
    """Apply nucleus (top-p) filtering to logits.

    Args:
        logits: Logits tensor
        p: Cumulative probability threshold

    Returns:
        Filtered logits
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Keep at least one token
    sorted_indices_to_remove[..., 0] = False

    # Scatter back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )

    return logits.masked_fill(indices_to_remove, float("-inf"))


# --------------------------------------------------------------------------
# torch utilities
# --------------------------------------------------------------------------


def _mps_available() -> bool:
    """Return True if MPS is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_optimal_device(
    force: Optional[str] = None,
) -> Tuple[torch.device, str, torch.dtype]:
    """Return best available accelerator (device, device_type, amp_dtype).

    The function tries CUDA → MPS → CPU, unless the user forces a choice via
    the ``force`` argument or the ``FORCE_DEVICE`` environment variable. The
    return value is intentionally simple—a tuple that works well with tuple
    unpacking in training scripts.
    """

    def _normalize(device_str: str) -> str:
        return device_str.split(":", 1)[0]

    requested = (force or os.getenv("FORCE_DEVICE", "")).strip().lower()
    valid_types = {"cuda", "mps", "cpu"}

    if requested:
        requested_type = _normalize(requested)
        if requested_type not in valid_types:
            print(
                f"Warning: unsupported FORCE_DEVICE='{requested}'. "
                "Falling back to auto-detect."
            )
            requested = ""
        elif requested_type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back.")
            requested = ""
        elif requested_type == "mps" and not _mps_available():
            print("Warning: MPS requested but not available, falling back.")
            requested = ""

    if requested:
        try:
            device = torch.device(requested)
        except (RuntimeError, ValueError) as err:
            print(f"Warning: could not create device '{requested}' ({err}).")
            requested = ""
        else:
            device_type = _normalize(requested)
            if device_type == "cuda":
                index = device.index or 0
                device_count = torch.cuda.device_count()
                if index >= device_count:
                    print(
                        f"Warning: CUDA index {index} unavailable "
                        f"(found {device_count} device(s)). Falling back."
                    )
                    requested = ""
                else:
                    name = torch.cuda.get_device_name(index)
                    print(f"Using CUDA device {index}: {name}")
                    return device, "cuda", torch.bfloat16
            elif device_type == "mps":
                print("Using Apple Silicon (MPS)")
                return device, "mps", torch.bfloat16
            else:
                print("Using CPU (forced)")
                return device, "cpu", torch.bfloat16

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"Using CUDA: {name}")
        return device, "cuda", torch.bfloat16

    if _mps_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS)")
        return device, "mps", torch.bfloat16

    device = torch.device("cpu")
    print("Using CPU (no GPU acceleration available)")
    return device, "cpu", torch.bfloat16


@dataclass
class _LayerSummary:
    """Summary statistics for a single layer in the model."""

    name: str
    param_shape: Optional[torch.Size]
    inclusive_total_params: int
    inclusive_trainable_params: int


def model_summary(
    model: nn.Module,
    max_depth: int = 4,
    show_param_shapes: bool = False,
    show_frozen_breakdown: bool = False,
) -> None:
    """Print hierarchical summary of model with parameter counts.

    :param model: PyTorch model to summarize
    :param max_depth: Maximum depth of hierarchy to display
    :param show_param_shapes: Whether to show parameter shapes
    :param show_frozen_breakdown: If True, display separate trainable/frozen counts
        per module. Defaults to False for a simpler view that highlights whether
        a module is fully trainable, fully frozen, or mixed.
    """

    # ---------- formatting helpers ----------
    def _format_number(num: int) -> str:
        return f"{num:,}" if num > 0 else "--"

    def _format_shape(shape: Optional[torch.Size]) -> str:
        return "x".join(map(str, shape)) if shape else "N/A"

    # ---------- build param info once ----------
    # Map: id(param) -> (numel, requires_grad)
    param_info: Dict[int, Tuple[int, bool]] = {}
    for p in model.parameters(recurse=True):
        pid = id(p)
        if pid not in param_info:
            param_info[pid] = (p.numel(), bool(p.requires_grad))

    # Fast path: totals only
    if max_depth <= 0:
        total_params = sum(n for (n, _) in param_info.values())
        trainable_params = sum(n for (n, rg) in param_info.values() if rg)
        print("=" * 50)
        print("Total params:", _format_number(total_params))
        print("Trainable params:", _format_number(trainable_params))
        nontrain = total_params - trainable_params
        print("Non-trainable params:", _format_number(nontrain))
        print("=" * 50)
        return

    summary_list: List[_LayerSummary] = []

    def summarize_recursive(module: nn.Module, depth: int, prefix: str) -> Set[int]:
        """Recursively build summary for module subtree.

        :param module: Current module being processed
        :param depth: Current depth in hierarchy
        :param prefix: Indentation prefix for display
        :return: Set of unique parameter IDs in subtree
        """
        # If we're beyond the print depth, just return the deduped set upward
        if depth > max_depth:
            ids = {id(p) for p in module.parameters(recurse=True)}
            return ids

        # Direct parameters of *this* module (non-recursive)
        direct_ids: Set[int] = {id(p) for p in module.parameters(recurse=False)}

        # Recurse into children and union their sets
        child_ids: Set[int] = set()
        for child in module.children():
            child_ids |= summarize_recursive(child, depth + 1, prefix + "  ")

        all_ids = direct_ids | child_ids

        # Inclusive counts from the deduped set
        total = sum(param_info[i][0] for i in all_ids)
        trainable = sum(param_info[i][0] for i in all_ids if param_info[i][1])

        # First direct trainable parameter shape (display purpose only)
        param_shape = next(
            (p.shape for p in module.parameters(recurse=False) if p.requires_grad),
            None,
        )

        summary_list.append(
            _LayerSummary(
                name=f"{prefix}{type(module).__name__}",
                param_shape=param_shape,
                inclusive_total_params=total,
                inclusive_trainable_params=trainable,
            )
        )
        return all_ids

    # Build the list (pre-order traversal)
    summarize_recursive(model, 1, "")

    # Totals from the whole model (already deduped)
    total_params = sum(n for (n, _) in param_info.values())
    trainable_params = sum(n for (n, rg) in param_info.values() if rg)

    # ---------- printing ----------
    name_col_width = max(len("Layer (type)"), max(len(s.name) for s in summary_list))
    shape_col_width = 0
    if show_param_shapes:
        shape_col_width = max(
            len("Param Shape"),
            max(len(_format_shape(s.param_shape)) for s in summary_list),
        )

    params_col_width = max(
        len("Param #"),
        max(len(_format_number(s.inclusive_total_params)) for s in summary_list),
    )

    header_parts = [f"{'Layer (type)':<{name_col_width}}"]
    if show_param_shapes:
        header_parts.append(f"{'Param Shape':>{shape_col_width}}")

    header_parts.append(f"{'Param #':>{params_col_width}}")

    if show_frozen_breakdown:
        trainable_col_width = max(
            len("Trainable #"),
            max(
                len(_format_number(s.inclusive_trainable_params)) for s in summary_list
            ),
        )
        frozen_col_width = max(
            len("Frozen #"),
            max(
                len(
                    _format_number(
                        s.inclusive_total_params - s.inclusive_trainable_params
                    )
                )
                for s in summary_list
            ),
        )
        header_parts.append(f"{'Trainable #':>{trainable_col_width}}")
        header_parts.append(f"{'Frozen #':>{frozen_col_width}}")
    else:

        def _grad_state(total: int, trainable: int) -> str:
            if trainable == 0:
                return "frozen"
            if trainable == total:
                return "trainable"
            return "mixed"

        grad_states = [
            _grad_state(
                s.inclusive_total_params,
                s.inclusive_trainable_params,
            )
            for s in summary_list
        ]
        grad_state_width = max(
            len("Grad State"), max(len(state) for state in grad_states)
        )
        header_parts.append(f"{'Grad State':>{grad_state_width}}")

    col_spacing = "  "

    header = col_spacing.join(header_parts)
    sep = "=" * len(header)

    print(sep)
    print(header)
    print(sep)
    for e in summary_list:
        parts = [f"{e.name:<{name_col_width}}"]
        if show_param_shapes:
            parts.append(f"{_format_shape(e.param_shape):>{shape_col_width}}")
        parts.append(f"{_format_number(e.inclusive_total_params):>{params_col_width}}")
        if show_frozen_breakdown:
            parts.append(
                f"{_format_number(e.inclusive_trainable_params):>{trainable_col_width}}"
            )
            frozen = e.inclusive_total_params - e.inclusive_trainable_params
            parts.append(f"{_format_number(frozen):>{frozen_col_width}}")
        else:
            state = _grad_state(e.inclusive_total_params, e.inclusive_trainable_params)
            parts.append(f"{state:>{grad_state_width}}")
        print(col_spacing.join(parts))
    print(sep)
    print(f"Total params: {_format_number(total_params)}")
    print(f"Trainable params: {_format_number(trainable_params)}")
    print(f"Non-trainable params: {_format_number(total_params - trainable_params)}")
    print(sep)
