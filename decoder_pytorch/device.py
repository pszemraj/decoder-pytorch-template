# decoder_pytorch/device.py

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch


__all__ = ["DeviceSelection", "get_optimal_device"]


@dataclass(frozen=True)
class DeviceSelection:
    """Result of device selection."""

    device: torch.device  # For .to() / model placement
    device_type: str  # For torch.autocast(device_type=...)
    device_info: str  # Human-readable description
    amp_dtype: torch.dtype  # Suggested autocast dtype (bfloat16 on accel)


def _cuda_info(index: int) -> str:
    name = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    return f"CUDA GPU: {name} (compute {major}.{minor})"


def _mps_available() -> bool:
    # torch.backends.mps is present only on macOS builds
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_optimal_device(
    preferred_order: Iterable[str] = ("cuda", "mps", "cpu"),
    device_index: int = 0,
    *,
    force: Optional[str] = None,
    enable_tf32: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> DeviceSelection:
    """
    Detect and return the best available device for PyTorch.

    Priority is defined by `preferred_order` (default: CUDA > MPS > CPU).
    You can force a specific device via `force="cuda"|"mps"|"cpu"` or the
    environment variable FORCE_DEVICE with the same values.

    Returns:
        DeviceSelection(device, device_type, device_info, amp_dtype)

    Notes:
        - For CUDA & MPS, the suggested autocast dtype is torch.bfloat16.
        - CPU autocast is limited and typically uses float32.
        - If `enable_tf32` is provided, it toggles TF32 on CUDA (Ampere+).
    """
    log = logger or logging.getLogger(__name__)
    choice = (force or os.getenv("FORCE_DEVICE", "")).lower().strip()

    # Normalize and validate preference list
    normalized_order: Tuple[str, ...] = (
        tuple(
            d
            for d in (choice,) + tuple(preferred_order)
            if d in ("cuda", "mps", "cpu") and d
        )
        if choice
        else tuple(d for d in preferred_order if d in ("cuda", "mps", "cpu"))
    )

    # Try each device in order
    for dev_type in normalized_order:
        if dev_type == "cuda" and torch.cuda.is_available():
            # Optional CUDA knobs
            if enable_tf32 is not None:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
                    torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
                except Exception:
                    # TF32 toggles aren't critical-ignore if unavailable
                    pass

            try:
                info = _cuda_info(device_index)
            except Exception as e:
                info = f"CUDA GPU (info unavailable: {e})"

            device = torch.device("cuda", index=device_index)
            amp_dtype = torch.bfloat16
            log.info("Using %s", info)
            return DeviceSelection(device, "cuda", info, amp_dtype)

        if dev_type == "mps" and _mps_available():
            info = "Apple Silicon (MPS)"
            device = torch.device("mps")
            amp_dtype = torch.bfloat16
            log.info("Using %s", info)
            return DeviceSelection(device, "mps", info, amp_dtype)

        if dev_type == "cpu":
            info = "CPU"
            device = torch.device("cpu")
            amp_dtype = torch.float32
            log.info("Using %s (no GPU acceleration available)", info)
            return DeviceSelection(device, "cpu", info, amp_dtype)

    # Absolute fallback (shouldn't be reached)
    info = "CPU"
    device = torch.device("cpu")
    amp_dtype = torch.float32
    log.warning("Falling back to CPU; no preferred devices were available.")
    return DeviceSelection(device, "cpu", info, amp_dtype)
