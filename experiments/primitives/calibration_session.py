#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, replace
import math
import time

from src.backends import run_backend
from src.calibration import (
    run_calibration,
    resolve_cpu_batch_size_candidates,
    resolve_cpu_worker_candidates,
)
from src.schemas import GridSpec, HybridMode, MLTaskSpec, SchedulerRequest, VanillaMode


def measure_runtime(request: SchedulerRequest, **runtime_kwargs):
    start = time.perf_counter()
    run_backend(request, **runtime_kwargs)
    return time.perf_counter() - start


def derive_calibration_grid_resolution(cpu_worker_values):
    required_points = 4 * (1 + max(cpu_worker_values))
    return max(4, math.ceil(math.sqrt(required_points)))


def _run_without_cache_session(
    variant_tasks: list[MLTaskSpec],
    calibration_grid: GridSpec,
    grid: GridSpec,
    gpu_batch_size: int,
    cpu_worker_values: tuple[int, ...],
    cpu_batch_sizes: tuple[int, ...],
    retry: int,
    seed: int,
    gpu_slowdown_factor: float,
) -> dict:
    """
    Empirical without-cache session: each variant pays full baseline + calibration
    overhead before its grid execution. Records actual wall time per variant.
    """
    session_start = time.perf_counter()
    variant_details = []

    for index, task in enumerate(variant_tasks):
        variant_seed = seed + index

        baseline_start = time.perf_counter()
        baseline_result = run_backend(
            SchedulerRequest(task, calibration_grid, VanillaMode(gpu_batch_size)),
            seed=variant_seed,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
        baseline_s = time.perf_counter() - baseline_start

        calibration_start = time.perf_counter()
        execution_mode, best_hybrid_s = run_calibration(
            SchedulerRequest(task, calibration_grid, HybridMode(gpu_batch_size)),
            baseline_result.record.measurement.total_s,
            cpu_worker_values,
            cpu_batch_sizes,
            retry,
            seed=variant_seed,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
        calibration_s = time.perf_counter() - calibration_start

        grid_start = time.perf_counter()
        run_backend(
            SchedulerRequest(task, grid, execution_mode),
            seed=variant_seed,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
        grid_s = time.perf_counter() - grid_start

        variant_details.append(
            {
                "baseline_s": round(baseline_s, 3),
                "calibration_s": round(calibration_s, 3),
                "grid_s": round(grid_s, 3),
                "execution_mode": asdict(execution_mode),
                "best_hybrid_s": round(best_hybrid_s, 6) if best_hybrid_s is not None else None,
                "best_hybrid_minus_gpu_only_s": (
                    round(best_hybrid_s - baseline_result.record.measurement.total_s, 6)
                    if best_hybrid_s is not None else None
                ),
            }
        )

    session_total_s = time.perf_counter() - session_start
    return {
        "session_total_s": round(session_total_s, 3),
        "baseline_s": round(sum(v["baseline_s"] for v in variant_details), 3),
        "calibration_s": round(sum(v["calibration_s"] for v in variant_details), 3),
        "execution_s": round(sum(v["grid_s"] for v in variant_details), 3),
        "variants": len(variant_details),
        "variant_details": variant_details,
    }


def main(
    gpu_slowdown_factor: float,
    retry: int,
    model_variant_checkpoints: list[str],
    *,
    base_workload: MLTaskSpec,
    grid: GridSpec,
    gpu_batch_size: int,
    seed: int,
    measure_without_cache: bool = False,
):
    cpu_worker_values = resolve_cpu_worker_candidates()
    cpu_batch_sizes = resolve_cpu_batch_size_candidates(
        base_workload.dataset.sample_count,
        gpu_batch_size,
    )
    calibration_grid_resolution = derive_calibration_grid_resolution(cpu_worker_values)
    calibration_grid = GridSpec(calibration_grid_resolution, grid.scale)

    variant_tasks = [
        replace(base_workload, checkpoint_path=checkpoint_path)
        for checkpoint_path in model_variant_checkpoints
    ]
    calibration_task = variant_tasks[0]

    baseline_start = time.perf_counter()
    baseline_result = run_backend(
        SchedulerRequest(
            calibration_task,
            calibration_grid,
            VanillaMode(gpu_batch_size),
        ),
        seed=seed,
        gpu_slowdown_factor=gpu_slowdown_factor,
    )
    baseline_s = time.perf_counter() - baseline_start

    calibration_start = time.perf_counter()
    execution_mode, best_hybrid_s = run_calibration(
        SchedulerRequest(
            calibration_task,
            calibration_grid,
            HybridMode(gpu_batch_size),
        ),
        baseline_result.record.measurement.total_s,
        cpu_worker_values,
        cpu_batch_sizes,
        retry,
        seed=seed,
        gpu_slowdown_factor=gpu_slowdown_factor,
    )
    calibration_s = time.perf_counter() - calibration_start

    grid_processing_runtimes = [
        measure_runtime(
            SchedulerRequest(task, grid, execution_mode),
            seed=seed + index,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
        for index, task in enumerate(variant_tasks)
    ]

    execution_s = sum(grid_processing_runtimes)
    variants = len(variant_tasks)
    with_cache_session = {
        "baseline_s": round(baseline_s, 3),
        "calibration_s": round(calibration_s, 3),
        "execution_mode": asdict(execution_mode),
        "execution_s": round(execution_s, 3),
        "first_variant_total_s": round(
            baseline_s + calibration_s + grid_processing_runtimes[0], 3
        ),
        "subsequent_variants_total_s": round(sum(grid_processing_runtimes[1:]), 3),
        "session_total_s": round(baseline_s + calibration_s + execution_s, 3),
        "best_calibration_hybrid_s": round(best_hybrid_s, 6) if best_hybrid_s is not None else None,
        "best_hybrid_minus_gpu_only_s": (
            round(best_hybrid_s - baseline_result.record.measurement.total_s, 6)
            if best_hybrid_s is not None else None
        ),
    }

    if measure_without_cache:
        without_cache_session = _run_without_cache_session(
            variant_tasks,
            calibration_grid,
            grid,
            gpu_batch_size,
            cpu_worker_values,
            cpu_batch_sizes,
            retry,
            seed=seed,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
        without_cache_session["source"] = "empirical"
    else:
        without_cache_session = {
            "source": "derived",
            "variants": variants,
            "baseline_s": round(baseline_s * variants, 3),
            "calibration_s": round(calibration_s * variants, 3),
            "execution_s": round(execution_s, 3),
            "session_total_s": round(
                baseline_s * variants + calibration_s * variants + execution_s, 3
            ),
        }

    return {
        "setup": {
            "workload": base_workload.name,
            "checkpoint_path": base_workload.checkpoint_path,
            "gpu_slowdown_factor": gpu_slowdown_factor,
            "calibration_grid_resolution": calibration_grid_resolution,
            "execution_grid_resolution": grid.resolution,
            "cpu_worker_values": list(cpu_worker_values),
            "cpu_batch_sizes": list(cpu_batch_sizes),
            "retry": retry,
            "model_variant_checkpoints": model_variant_checkpoints,
            "measure_without_cache": measure_without_cache,
        },
        "with_cache_session": with_cache_session,
        "without_cache_session": without_cache_session,
    }
