from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class SectionTimings:
    perturbation_s: float = 0.0
    binding_s: float = 0.0
    batch_eval_s: float = 0.0
    total_grid_s: float = 0.0


@dataclass(frozen=True)
class TimingSample:
    started_at_s: float
    ended_at_s: float
    elapsed_s: float


@dataclass(frozen=True)
class CudaMemorySnapshot:
    available: bool
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    max_allocated_bytes: int | None = None
    max_reserved_bytes: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ProcessMemorySnapshot:
    available: bool
    rss_bytes: int | None = None
    reason: str | None = None


def time_block(started_at_s: float | None = None) -> TimingSample:
    started = perf_counter() if started_at_s is None else started_at_s
    ended = perf_counter()
    return TimingSample(
        started_at_s=started,
        ended_at_s=ended,
        elapsed_s=ended - started,
    )


def reset_cuda_peak_memory(device: Any | None = None) -> bool:
    torch = _import_torch()
    if torch is None or not _cuda_available(torch):
        return False
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return False
    return True


def cuda_memory_snapshot(device: Any | None = None) -> CudaMemorySnapshot:
    torch = _import_torch()
    if torch is None:
        return CudaMemorySnapshot(False, reason="torch is unavailable")
    if not _cuda_available(torch):
        return CudaMemorySnapshot(False, reason="cuda is unavailable")

    try:
        return CudaMemorySnapshot(
            available=True,
            allocated_bytes=int(torch.cuda.memory_allocated(device)),
            reserved_bytes=int(torch.cuda.memory_reserved(device)),
            max_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            max_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
    except Exception as exc:
        return CudaMemorySnapshot(False, reason=f"{type(exc).__name__}: {exc}")


def process_memory_snapshot() -> ProcessMemorySnapshot:
    try:
        import resource
    except Exception as exc:
        return ProcessMemorySnapshot(False, reason=f"{type(exc).__name__}: {exc}")

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception as exc:
        return ProcessMemorySnapshot(False, reason=f"{type(exc).__name__}: {exc}")

    rss = int(usage.ru_maxrss)
    if _is_darwin():
        rss_bytes = rss
    else:
        rss_bytes = rss * 1024
    return ProcessMemorySnapshot(True, rss_bytes=rss_bytes)


def _import_torch() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch


def _cuda_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return False
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _is_darwin() -> bool:
    try:
        import sys

        return sys.platform == "darwin"
    except Exception:
        return False
