#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import time

from src.backends import run_backend
from src.calibration import run_calibration
from src.config import (
    HybridExecutionConfig,
    VanillaExecutionConfig,
    load_config,
)
from src.results import load_cached_run_summary
from src.runner import run_baseline_and_persist


def derive_vanilla_workload(workload):
    baseline = workload.clone()
    baseline.backend = "vanilla"
    baseline.resources.cpu_workers = 0
    baseline.data.cpu_batch_size = None
    return baseline


def measure_runtime(execution_config):
    start = time.perf_counter()
    run_backend(execution_config)
    return time.perf_counter() - start


def build_variant_workload(base_config, checkpoint_path: str, variant_index: int):
    workload = base_config.clone()
    workload.model.checkpoint_path = checkpoint_path
    workload.seed = base_config.seed + variant_index
    return workload


def derive_calibration_workload(workload, grid_resolution: int):
    calibration_workload = workload.clone()
    calibration_workload.grid.resolution = grid_resolution
    return calibration_workload


def derive_calibration_grid_resolution(cpu_worker_values):
    required_points = 4 * (1 + max(cpu_worker_values))
    return max(4, math.ceil(math.sqrt(required_points)))


def resolve_cpu_worker_values():
    cpu_core_count = max(1, os.cpu_count() or 1)
    values = []
    candidate = 1

    while candidate < cpu_core_count:
        values.append(candidate)
        candidate *= 2

    if not values or values[-1] != cpu_core_count:
        values.append(cpu_core_count)

    return tuple(values)


def resolve_cpu_batch_sizes(workload):
    upper_bound = max(1, min(workload.data.batch_size, workload.data.subset_size))
    values = []
    candidate = 4

    while candidate < upper_bound:
        values.append(candidate)
        candidate *= 2

    if not values or values[-1] != upper_bound:
        values.append(upper_bound)

    return tuple(values)


def main(
    gpu_slowdown_factor: float,
    retry: int,
    model_variant_checkpoints: list[str],
    hybrid_config_path: str,
):
    base_config = load_config(hybrid_config_path)
    base_config.runtime.gpu_slowdown_factor = gpu_slowdown_factor
    cpu_worker_values = resolve_cpu_worker_values()
    cpu_batch_sizes = resolve_cpu_batch_sizes(base_config)
    calibration_grid_resolution = derive_calibration_grid_resolution(cpu_worker_values)

    workloads = [
        build_variant_workload(base_config, checkpoint_path, index)
        for index, checkpoint_path in enumerate(model_variant_checkpoints)
    ]

    calibration_workload = derive_calibration_workload(
        workloads[0], calibration_grid_resolution
    )

    baseline_start = time.perf_counter()
    baseline_workload = derive_vanilla_workload(calibration_workload)
    try:
        baseline_summary = load_cached_run_summary(baseline_workload)
    except FileNotFoundError:
        baseline_summary = run_baseline_and_persist(baseline_workload).record
    baseline_s = time.perf_counter() - baseline_start

    calibration_start = time.perf_counter()
    execution_policy = run_calibration(
        calibration_workload,
        baseline_summary.measurement.total_s,
        cpu_worker_values,
        cpu_batch_sizes,
        retry,
    )
    calibration_s = time.perf_counter() - calibration_start

    grid_processing_runtimes = []
    # note: each variant's workload differs in the model params value
    for workload in workloads:
        execution_config = (
            VanillaExecutionConfig(workload=derive_vanilla_workload(workload))
            if execution_policy["_tag"] == "vanilla"
            else HybridExecutionConfig(
                workload,
                cpu_workers=int(execution_policy["cpu"]["workers"]),
                cpu_batch_size=int(execution_policy["cpu"]["batch_size"]),
            )
        )
        grid_processing_runtimes.append(measure_runtime(execution_config))

    execution_s = sum(grid_processing_runtimes)
    variants = len(workloads)

    return {
        "setup": {
            "hybrid_config_path": hybrid_config_path,
            "gpu_slowdown_factor": gpu_slowdown_factor,
            "calibration_grid_resolution": calibration_grid_resolution,
            "execution_grid_resolution": base_config.grid.resolution,
            "cpu_worker_values": cpu_worker_values,
            "cpu_batch_sizes": cpu_batch_sizes,
            "retry": retry,
            "model_variant_checkpoints": model_variant_checkpoints,
        },
        "with_cache_session": (
            with_cache_session := {
                "baseline_s": baseline_s,
                "calibration_s": calibration_s,
                "execution_policy": execution_policy,
                "execution_s": execution_s,
                "first_variant_total_s": baseline_s
                + calibration_s
                + grid_processing_runtimes[0],
                "subsequent_variants_total_s": sum(grid_processing_runtimes[1:]),
                "session_total_s": baseline_s + calibration_s + execution_s,
            }
        ),
        "without_cache_session": {
            "variants": variants,
            "baseline_s": with_cache_session["baseline_s"] * variants,
            "calibration_s": with_cache_session["calibration_s"] * variants,
            "execution_s": with_cache_session["execution_s"],
            "session_total_s": (
                with_cache_session["baseline_s"] * variants
                + with_cache_session["calibration_s"] * variants
                + with_cache_session["execution_s"]
            ),
        },
    }
