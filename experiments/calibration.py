from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch

from experiments.candidates import hybrid
from experiments.candidates.base import GpuCandidate
from experiments.schemas import GridSpec, MLTaskSpec


@dataclass(frozen=True)
class CalibratedCell:
    selected_policy: Literal["gpu_only", "gpu_cpu_hybrid"]
    gpu_batch_size: int
    cpu_batch_size: int | None
    cpu_workers: int | None
    calibration_s: float
    baseline_total_s: float
    selected_total_s: float | None


def available_cpu_cores() -> int:
    return max(1, os.cpu_count() or 1)


def cpu_worker_candidates(max_workers: int | None = None) -> tuple[int, ...]:
    upper = available_cpu_cores() if max_workers is None else min(int(max_workers), available_cpu_cores())
    upper = max(1, upper)
    values: list[int] = []
    candidate = 1
    while candidate < upper:
        values.append(candidate)
        candidate *= 2
    if not values or values[-1] != upper:
        values.append(upper)
    return tuple(values)


def cpu_batch_size_candidates(sample_count: int, gpu_batch_size: int) -> tuple[int, ...]:
    upper = max(1, min(int(sample_count), int(gpu_batch_size)))
    if upper <= 4:
        return (upper,)
    values: list[int] = []
    candidate = 4
    while candidate < upper:
        values.append(candidate)
        candidate *= 2
    if not values or values[-1] != upper:
        values.append(upper)
    return tuple(values)


def calibrate(
    task: MLTaskSpec,
    grid: GridSpec,
    *,
    gpu_batch_size: int,
    baseline_total_s: float,
    cpu_workers: tuple[int, ...],
    cpu_batch_sizes: tuple[int, ...],
    patience: int,
    device: torch.device,
    seed: int,
    gpu_slowdown_factor: float = 1.0,
    gpu_candidate: GpuCandidate = GpuCandidate.baseline(),
) -> CalibratedCell:
    """Patience-bounded B-cell sweep [ansel2014opentuner].

    Stops after `patience` consecutive non-improvements over the rolling minimum.
    If no hybrid combination beats `baseline_total_s`, returns a `gpu_only` cell.
    """
    calibration_s = 0.0
    best_total: float | None = None
    best_workers: int | None = None
    best_batch: int | None = None
    rolling_min = baseline_total_s
    consecutive_non_improvements = 0
    early_stop = False

    for workers in cpu_workers:
        if early_stop:
            break
        best_for_workers: float | None = None
        for batch in cpu_batch_sizes:
            result = hybrid.run(
                task, grid,
                gpu_batch_size=gpu_batch_size,
                cpu_batch_size=batch,
                cpu_workers=workers,
                device=device, seed=seed,
                gpu_slowdown_factor=gpu_slowdown_factor,
                gpu_candidate=gpu_candidate,
            )
            calibration_s += float(result.total_grid_s)
            total = float(result.total_grid_s)
            if best_for_workers is None or total < best_for_workers:
                best_for_workers = total
            if total < rolling_min:
                rolling_min = total
                best_total = total
                best_workers = workers
                best_batch = batch
                consecutive_non_improvements = 0
                continue
            consecutive_non_improvements += 1
            if consecutive_non_improvements >= patience:
                early_stop = True
                break

    if best_total is None or best_total >= baseline_total_s:
        return CalibratedCell(
            selected_policy="gpu_only",
            gpu_batch_size=int(gpu_batch_size),
            cpu_batch_size=None,
            cpu_workers=None,
            calibration_s=calibration_s,
            baseline_total_s=baseline_total_s,
            selected_total_s=None,
        )
    return CalibratedCell(
        selected_policy="gpu_cpu_hybrid",
        gpu_batch_size=int(gpu_batch_size),
        cpu_batch_size=int(best_batch),
        cpu_workers=int(best_workers),
        calibration_s=calibration_s,
        baseline_total_s=baseline_total_s,
        selected_total_s=best_total,
    )
