from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from experiments.candidates import hybrid
from experiments.candidates.base import GpuCandidate
from experiments.cpu_resources import cpu_worker_candidates
from experiments.schemas import GridSpec, MLTaskSpec


@dataclass(frozen=True)
class CompositionSelection:
    selected_path: Literal["gpu_only", "gpu_cpu_hybrid"]
    gpu_batch_size: int
    cpu_batch_size: int | None
    cpu_workers: int | None
    selection_probe_s: float
    baseline_total_s: float
    selected_total_s: float | None
    max_hybrid_cpu_points: int = 0
    selection_trials: tuple[dict[str, Any], ...] = field(default_factory=tuple)


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


def select_composition(
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
) -> CompositionSelection:
    """Patience-bounded native composition selection probe.

    Stops after `patience` consecutive non-improvements over the rolling minimum.
    If no hybrid combination beats `baseline_total_s`, selects `gpu_only`.
    """
    selection_probe_s = 0.0
    best_total: float | None = None
    best_workers: int | None = None
    best_batch: int | None = None
    rolling_min = baseline_total_s
    consecutive_non_improvements = 0
    early_stop = False
    max_hybrid_cpu_points = 0
    selection_trials: list[dict[str, Any]] = []

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
            spawn_inclusive_total_s = float(result.total_grid_s)
            selection_total_s = _steady_state_selection_total_s(result)
            selection_probe_s += spawn_inclusive_total_s
            split = result.worker_throughput_split or {}
            max_hybrid_cpu_points = max(
                max_hybrid_cpu_points, int(split.get("cpu_points", 0) or 0)
            )
            selection_trials.append(
                {
                    "cpu_workers": int(workers),
                    "cpu_batch_size": int(batch),
                    "spawn_inclusive_total_s": spawn_inclusive_total_s,
                    "steady_state_selection_total_s": selection_total_s,
                    "cpu_points": int(split.get("cpu_points", 0) or 0),
                    "gpu_points": int(split.get("gpu_points", 0) or 0),
                    "cpu_max_wall_s": float(split.get("cpu_max_wall_s", 0.0) or 0.0),
                    "gpu_wall_s": float(split.get("gpu_wall_s", 0.0) or 0.0),
                }
            )
            total = selection_total_s
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
        return CompositionSelection(
            selected_path="gpu_only",
            gpu_batch_size=int(gpu_batch_size),
            cpu_batch_size=None,
            cpu_workers=None,
            selection_probe_s=selection_probe_s,
            baseline_total_s=baseline_total_s,
            selected_total_s=None,
            max_hybrid_cpu_points=max_hybrid_cpu_points,
            selection_trials=tuple(selection_trials),
        )
    return CompositionSelection(
        selected_path="gpu_cpu_hybrid",
        gpu_batch_size=int(gpu_batch_size),
        cpu_batch_size=int(best_batch),
        cpu_workers=int(best_workers),
        selection_probe_s=selection_probe_s,
        baseline_total_s=baseline_total_s,
        selected_total_s=best_total,
        max_hybrid_cpu_points=max_hybrid_cpu_points,
        selection_trials=tuple(selection_trials),
    )


def _steady_state_selection_total_s(result) -> float:
    split = result.worker_throughput_split or {}
    worker_walls = [
        float(split.get("cpu_max_wall_s", 0.0) or 0.0),
        float(split.get("gpu_wall_s", 0.0) or 0.0),
    ]
    steady_state_s = max(worker_walls)
    if steady_state_s > 0.0:
        return steady_state_s
    return float(result.total_grid_s)
