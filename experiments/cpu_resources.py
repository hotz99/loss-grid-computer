from __future__ import annotations

import os


def available_cpu_cores() -> int:
    return max(1, os.cpu_count() or 1)


def max_cpu_workers(max_workers: int | None = None) -> int:
    """Worker count p_max used when no sweep is run (RQ2 fixes p=p_max)."""
    upper = (
        available_cpu_cores()
        if max_workers is None
        else min(int(max_workers), available_cpu_cores())
    )
    return max(1, upper)


def cpu_worker_candidates(max_workers: int | None = None) -> tuple[int, ...]:
    upper = max_cpu_workers(max_workers)
    values: list[int] = []
    candidate = 1
    while candidate < upper:
        values.append(candidate)
        candidate *= 2
    if not values or values[-1] != upper:
        values.append(upper)
    return tuple(values)
