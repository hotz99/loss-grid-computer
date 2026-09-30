from __future__ import annotations

import sys

import torch


def reset_peak_counters(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass


def peak_cuda_bytes(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated(device))
    except Exception:
        return None


def peak_cpu_bytes() -> int | None:
    try:
        import resource
    except Exception:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:
        return None
    rss = int(usage.ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024
