"""Timing helpers with optional CUDA synchronization."""

import time

import torch


def sync_if_needed(device=None):
    """Synchronize CUDA work before reading wall-clock time."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    if isinstance(device, str):
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    if isinstance(device, torch.device):
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()


def now(device=None):
    """Return a synchronized wall-clock timestamp in seconds."""
    sync_if_needed(device)
    return time.perf_counter()


def now_ns(device=None):
    """Return a synchronized wall-clock timestamp in nanoseconds."""
    sync_if_needed(device)
    return time.perf_counter_ns()


def offset_ns(base_ns, current_ns):
    """Convert an absolute monotonic timestamp to a request-relative offset."""
    return int(current_ns - base_ns)


def ns_to_ms(offset_ns_value):
    """Convert a relative nanosecond offset to milliseconds."""
    return max(float(offset_ns_value) / 1_000_000.0, 0.0)
