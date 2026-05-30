from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace

import torch

from experiments.calibration import CalibratedCell
from experiments.candidates import GpuCandidate, run_standalone
from experiments.candidates import baseline as baseline_candidate
from experiments.candidates import hybrid as hybrid_candidate
from experiments.candidates.base import CandidateRunOutput, Surface
from experiments.schemas import GridSpec, MLTaskSpec


Surface_T = Surface


@dataclass(frozen=True)
class CheckpointTime:
    checkpoint_path: str
    t_grid_s: float
    records: Surface_T


@dataclass(frozen=True)
class SessionRecord:
    per_checkpoint: tuple[CheckpointTime, ...]
    sum_per_checkpoint_s: float
    total_s: float
    mean_t_grid_s: float
    sigma_rel: float


def _at_checkpoint(task: MLTaskSpec, checkpoint_path: str) -> MLTaskSpec:
    return replace(task, checkpoint_path=checkpoint_path)


def _sigma_rel(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0.0:
        return 0.0
    return statistics.stdev(values) / mean


def _summarize(
    per_checkpoint: list[CheckpointTime],
    extra_s: float,
) -> SessionRecord:
    times = [item.t_grid_s for item in per_checkpoint]
    sum_per = float(sum(times))
    mean = statistics.mean(times) if times else 0.0
    return SessionRecord(
        per_checkpoint=tuple(per_checkpoint),
        sum_per_checkpoint_s=sum_per,
        total_s=sum_per + extra_s,
        mean_t_grid_s=mean,
        sigma_rel=_sigma_rel(times),
    )


def vanilla_session(
    task: MLTaskSpec,
    grid: GridSpec,
    checkpoints: tuple[str, ...],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    gpu_slowdown_factor: float = 1.0,
) -> SessionRecord:
    per: list[CheckpointTime] = []
    for path in checkpoints:
        result = baseline_candidate.run(
            _at_checkpoint(task, path), grid,
            batch_size=batch_size, device=device, seed=seed,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
        per.append(CheckpointTime(checkpoint_path=path, t_grid_s=result.total_grid_s, records=result.records))
    return _summarize(per, extra_s=0.0)


def _run_with_cell(
    task: MLTaskSpec,
    grid: GridSpec,
    *,
    cell: CalibratedCell,
    gpu_candidate: GpuCandidate,
    device: torch.device,
    seed: int,
    gpu_slowdown_factor: float = 1.0,
) -> CandidateRunOutput:
    if cell.selected_policy == "gpu_only":
        return run_standalone(
            gpu_candidate, task, grid,
            batch_size=cell.gpu_batch_size, device=device, seed=seed,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
    return hybrid_candidate.run(
        task, grid,
        gpu_batch_size=cell.gpu_batch_size,
        cpu_batch_size=int(cell.cpu_batch_size or 0),
        cpu_workers=int(cell.cpu_workers or 0),
        device=device, seed=seed,
        gpu_slowdown_factor=gpu_slowdown_factor,
        gpu_candidate=gpu_candidate,
    )


def cached_composed_session(
    task: MLTaskSpec,
    grid: GridSpec,
    checkpoints: tuple[str, ...],
    *,
    cell: CalibratedCell,
    gpu_candidate: GpuCandidate,
    device: torch.device,
    seed: int,
    gpu_slowdown_factor: float = 1.0,
    compile_s: float = 0.0,
) -> SessionRecord:
    """T_session = calibration_s + compile_s + sum(per-checkpoint T_grid).

    The cached one-time setup cost is calibration plus the A config's compile
    cold-start: both are paid once, cached, and reused across the session. The
    per-checkpoint T_grid is the warm steady-state time (compile is excluded
    from total_grid_s in every candidate path). compile_s is 0 for A configs
    that do not compile, so the model is uniform across workloads.
    """
    per: list[CheckpointTime] = []
    for path in checkpoints:
        result = _run_with_cell(
            _at_checkpoint(task, path), grid,
            cell=cell, gpu_candidate=gpu_candidate, device=device, seed=seed,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
        per.append(CheckpointTime(checkpoint_path=path, t_grid_s=result.total_grid_s, records=result.records))
    return _summarize(per, extra_s=cell.calibration_s + compile_s)


def break_even_n(t_v: float, t_p: float, one_time_s: float) -> int | None:
    """⌈one_time_s / (T_v − T_p)⌉ when T_v > T_p, else absent.

    one_time_s is the cached setup cost (calibration plus compile cold-start)."""
    if t_v <= t_p:
        return None
    return int(math.ceil(one_time_s / (t_v - t_p)))
