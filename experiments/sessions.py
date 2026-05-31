from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import torch
from torch.nn.utils import vector_to_parameters

from experiments import device as device_mod
from experiments.calibration import CalibratedCell
from experiments.candidates import GpuCandidate, make_chunk_evaluator, run_standalone
from experiments.candidates import baseline as baseline_candidate
from experiments.candidates import hybrid as hybrid_candidate
from experiments.candidates.base import CandidateRunOutput, Surface
from experiments.data import build_dataloader, build_dataset
from experiments.grid import build_direction_vectors, build_grid_points
from experiments.models import build_model, load_checkpoint
from experiments.schemas import GridSpec, MLTaskSpec


Surface_T = Surface


@dataclass(frozen=True)
class CheckpointTime:
    checkpoint_path: str
    t_grid_s: float
    records: Surface_T
    diagnostics: dict[str, Any] = field(default_factory=dict)


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
        per.append(
            CheckpointTime(
                checkpoint_path=path,
                t_grid_s=result.total_grid_s,
                records=result.records,
                diagnostics=result.diagnostics,
            )
        )
    return _summarize(per, extra_s=0.0)


def gpu_only_session(
    task: MLTaskSpec,
    grid: GridSpec,
    checkpoints: tuple[str, ...],
    *,
    gpu_candidate: GpuCandidate,
    batch_size: int,
    device: torch.device,
    seed: int,
    gpu_slowdown_factor: float = 1.0,
) -> SessionRecord:
    """Warm in-process GPU-side session.

    The evaluator is built and warmed once. Each checkpoint refreshes model
    weights and the captured base/direction tensors in place before timing the
    steady-state grid.
    """
    if not checkpoints:
        return _summarize([], extra_s=0.0)

    device_mod.seed_all(device, seed)
    model = build_model(_at_checkpoint(task, checkpoints[0])).to(device)
    model.eval()
    dataset = build_dataset(task, seed)
    data_loader = build_dataloader(
        dataset, batch_size, pin_memory=(device.type == "cuda")
    )
    base_cpu, dir_a_cpu, dir_b_cpu = build_direction_vectors(model, seed)
    base = base_cpu.to(device)
    direction_a = dir_a_cpu.to(device)
    direction_b = dir_b_cpu.to(device)
    vector_to_parameters(base, model.parameters())
    device_mod.synchronize(device)
    evaluator = make_chunk_evaluator(
        gpu_candidate,
        model=model,
        data_loader=data_loader,
        device=device,
        task=task,
        base_device=base,
        direction_a_device=direction_a,
        direction_b_device=direction_b,
    )
    compile_cold_start_s = float(evaluator.warmup() or 0.0)
    device_mod.synchronize(device)
    points = build_grid_points(grid)

    per: list[CheckpointTime] = []
    for path in checkpoints:
        load_checkpoint(model, path)
        model.eval()
        base_cpu, dir_a_cpu, dir_b_cpu = build_direction_vectors(model, seed)
        base.copy_(base_cpu.to(device))
        direction_a.copy_(dir_a_cpu.to(device))
        direction_b.copy_(dir_b_cpu.to(device))
        vector_to_parameters(base, model.parameters())
        device_mod.synchronize(device)

        start = perf_counter()
        records = evaluator.evaluate(points)
        device_mod.synchronize(device)
        eval_elapsed = perf_counter() - start
        device_mod.apply_gpu_slowdown(device, gpu_slowdown_factor, eval_elapsed)
        device_mod.synchronize(device)
        total_grid_s = perf_counter() - start
        diagnostics = {
            **evaluator.diagnostics(),
            "candidate": gpu_candidate.name,
            "device": device.type,
            "compile_cold_start_s": compile_cold_start_s,
        }
        per.append(
            CheckpointTime(
                checkpoint_path=path,
                t_grid_s=total_grid_s,
                records=records,
                diagnostics=diagnostics,
            )
        )
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


def hybrid_pool_session(
    task: MLTaskSpec,
    grid: GridSpec,
    checkpoints: tuple[str, ...],
    *,
    cell: CalibratedCell,
    gpu_candidate: GpuCandidate,
    device: torch.device,
    seed: int,
    gpu_slowdown_factor: float = 1.0,
) -> tuple[SessionRecord, float]:
    if cell.selected_policy != "gpu_cpu_hybrid":
        raise ValueError("hybrid_pool_session requires a gpu_cpu_hybrid cell")
    with hybrid_candidate.HybridPool(
        task,
        grid,
        gpu_batch_size=cell.gpu_batch_size,
        cpu_batch_size=int(cell.cpu_batch_size or 0),
        cpu_workers=int(cell.cpu_workers or 0),
        device=device,
        seed=seed,
        gpu_slowdown_factor=gpu_slowdown_factor,
        gpu_candidate=gpu_candidate,
    ) as pool:
        per: list[CheckpointTime] = []
        for path in checkpoints:
            result = pool.run_grid(path)
            per.append(
                CheckpointTime(
                    checkpoint_path=path,
                    t_grid_s=result.total_grid_s,
                    records=result.records,
                    diagnostics={
                        **result.diagnostics,
                        "recompile_count": result.recompile_count,
                    },
                )
            )
        return _summarize(per, extra_s=0.0), pool.pool_startup_s


def break_even_n(t_v: float, t_p: float, one_time_s: float) -> int | None:
    """⌈one_time_s / (T_v − T_p)⌉ when T_v > T_p, else absent.

    one_time_s is the cached setup cost (calibration plus compile cold-start)."""
    if t_v <= t_p:
        return None
    return int(math.ceil(one_time_s / (t_v - t_p)))
