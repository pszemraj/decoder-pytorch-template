"""Llama-style transformer for language modeling experiments."""

from .device import DeviceSelection, get_optimal_device
from .llama import Llama
from .utils import (
    configure_tf32,
    gumbel_noise,
    gumbel_sample,
    log,
    min_p_filter,
    model_summary,
    top_k_filter,
    top_p_filter,
)

__all__ = [
    "Llama",
    # Sampling utilities
    "log",
    "gumbel_noise",
    "gumbel_sample",
    "min_p_filter",
    "top_k_filter",
    "top_p_filter",
    # Torch utilities
    "configure_tf32",
    "model_summary",
    # Device utilities
    "DeviceSelection",
    "get_optimal_device",
]
