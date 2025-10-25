# decoder_pytorch/device.py

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Iterable, Optional, Tuple

import torch


__all__ = ["DeviceSelection", "get_optimal_device"]


@dataclass(frozen=True)
class DeviceSelection:
    """Result of device selection."""

    device: torch.device  # For .to() / model placement
    device_type: str  # For torch.autocast(device_type=...)
    device_info: str  # Human-readable description
    amp_dtype: torch.dtype  # Suggested autocast dtype (bfloat16 on accel)

    def autocast_context(self, enabled: bool = True) -> ContextManager[None]:
        """
        Return a context manager for automatic mixed precision.

        Args:
            enabled: Whether autocast should be enabled. If False, a no-op
                context manager is returned.
        """
        if not enabled:
            return nullcontext()

        return torch.autocast(device_type=self.device_type, dtype=self.amp_dtype)


def _cuda_info(index: int) -> str:
    name = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    return f"CUDA GPU: {name} (compute {major}.{minor})"


def _mps_available() -> bool:
    # torch.backends.mps is present only on macOS builds
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _parse_device_request(request: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse a device preference string and optional index.

    Supported formats:
        "cuda", "cuda:1", "mps", "cpu"
    """
    if request is None:
        return None, None

    normalized = request.strip().lower()
    if not normalized:
        return None, None

    if ":" in normalized:
        base, index_str = normalized.split(":", 1)
        if base in ("cuda", "mps", "cpu"):
            try:
                return base, int(index_str)
            except ValueError:
                return base, None
        return None, None

    if normalized in ("cuda", "mps", "cpu"):
        return normalized, None

    return None, None


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
    environment variable FORCE_DEVICE with the same values. Suffix an index
    (e.g. "cuda:1") to target a specific accelerator.

    Returns:
        DeviceSelection(device, device_type, device_info, amp_dtype)

    Notes:
        - All devices (CUDA, MPS, CPU) use torch.bfloat16 for autocast.
        - bfloat16-compatible hardware is assumed (2025 AD standard).
        - If `enable_tf32` is provided, it toggles TF32 on CUDA (Ampere+).
        - Invalid entries in `preferred_order` are ignored with a debug log.
    """
    log = logger or logging.getLogger(__name__)
    force_raw = force or os.getenv("FORCE_DEVICE", "")
    forced_type, forced_index = _parse_device_request(force_raw)

    if forced_index is not None:
        device_index = forced_index
        log.info("Device override requested: %s:%s", forced_type, device_index)

    if forced_type is None and force_raw:
        log.warning(
            "FORCE_DEVICE=%s ignored; supported values are 'cuda', 'mps', or 'cpu'.",
            force_raw,
        )

    # Normalize and validate preference list (deduplicate while preserving order)
    candidate_order = []
    if forced_type:
        candidate_order.append(forced_type)

    for entry in preferred_order:
        parsed_type, _ = _parse_device_request(str(entry))
        if parsed_type:
            candidate_order.append(parsed_type)
        else:
            log.debug("Ignoring unknown device preference entry: %s", entry)

    if not candidate_order:
        candidate_order = ["cuda", "mps", "cpu"]

    seen = set()
    normalized_order: Tuple[str, ...] = tuple(
        dev for dev in candidate_order if not (dev in seen or seen.add(dev))
    )

    # Try each device in order
    for dev_type in normalized_order:
        if dev_type == "cuda" and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_index < 0 or device_index >= device_count:
                available_range = (
                    f"0-{device_count - 1}" if device_count else "no devices found"
                )
                log.warning(
                    "Requested CUDA device index %s is out of range (available: %s).",
                    device_index,
                    available_range,
                )
                continue

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
            amp_dtype = torch.bfloat16
            log.info("Using %s (no GPU acceleration available)", info)
            return DeviceSelection(device, "cpu", info, amp_dtype)

    # Absolute fallback (shouldn't be reached)
    if normalized_order:
        log.warning(
            "No preferred devices were available (requested order: %s). Falling back to CPU.",
            ", ".join(normalized_order),
        )
    else:
        log.warning("No device preferences provided. Falling back to CPU.")

    info = "CPU"
    device = torch.device("cpu")
    amp_dtype = torch.bfloat16
    return DeviceSelection(device, "cpu", info, amp_dtype)
