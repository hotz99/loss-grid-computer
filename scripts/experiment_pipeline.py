#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import statistics

from src.backends import run_backend
from src.calibration import run_calibration
from src.compare import compare_surfaces
from src.config import (
    DataConfig,
    ExperimentConfig,
    GridConfig,
    HybridExecutionConfig,
    MLTaskSpec,
    ResourcesConfig,
    RuntimeConfig,
    VanillaExecutionConfig,
)
from src.results import load_cached_run_with_surface, load_surface
from src.results import print_json
from src.runner import run_baseline_and_persist
from src.workloads import WORKLOADS
from scripts import experiment_2b, rq1_experiment


def run_experiments(
    task_spec: MLTaskSpec,
    *,
    vanilla_config: ExperimentConfig,
    hybrid_config: ExperimentConfig,
    repeats: int,
    max_slowdown: float,
    jump_factor: float,
    linear_samples: int,
    atol: float,
    rtol: float,
):
    hybrid_workload = hybrid_config.clone()
    hybrid_workload.task = task_spec

    vanilla_workload = vanilla_config.clone()
    vanilla_workload.task = task_spec

    try:
        vanilla_output_dir, vanilla_summary = load_cached_run_with_surface(
            vanilla_workload
        )
    except FileNotFoundError:
        vanilla_result = run_baseline_and_persist(vanilla_workload)
        vanilla_output_dir = vanilla_result.record.output_dir
        vanilla_summary = vanilla_result.record

    vanilla_surface = load_surface(vanilla_output_dir)

    rq1_summary = rq1_experiment.main(
        hybrid_workload=hybrid_workload,
        vanilla_surface=vanilla_surface,
        vanilla_total_s=vanilla_summary.measurement.total_s,
        repeats=repeats,
        max_slowdown=max_slowdown,
        jump_factor=jump_factor,
        linear_samples=linear_samples,
        atol=atol,
        rtol=rtol,
    )

    midpoint_slowdown = statistics.fmean(rq1_summary["crossover_region"])
    fixed_hybrid_workload = hybrid_workload.clone()
    fixed_hybrid_workload.runtime.gpu_slowdown_factor = midpoint_slowdown

    fixed_vanilla_workload = vanilla_workload.clone()
    fixed_vanilla_workload.runtime.gpu_slowdown_factor = midpoint_slowdown

    retry = 1
    cpu_worker_values = experiment_2b.resolve_cpu_worker_values()
    cpu_batch_sizes = experiment_2b.resolve_cpu_batch_sizes(fixed_hybrid_workload)
    selected_policy = run_calibration(
        fixed_hybrid_workload,
        vanilla_summary.measurement.total_s * midpoint_slowdown,
        cpu_worker_values,
        cpu_batch_sizes,
        retry,
    )

    fixed_vanilla_result = run_backend(
        VanillaExecutionConfig(workload=fixed_vanilla_workload)
    )
    fixed_hybrid_result = run_backend(
        VanillaExecutionConfig(workload=fixed_vanilla_workload)
        if selected_policy["_tag"] == "baseline"
        else HybridExecutionConfig(
            workload=fixed_hybrid_workload,
            cpu_workers=int(selected_policy["cpu"]["workers"]),
            cpu_batch_size=int(selected_policy["cpu"]["batch_size"]),
        )
    )
    comparison = compare_surfaces(
        lhs_surface=fixed_vanilla_result.records,
        rhs_surface=fixed_hybrid_result.records,
        atol=atol,
        rtol=rtol,
        lhs_total_s=fixed_vanilla_result.record.measurement.total_s,
        rhs_total_s=fixed_hybrid_result.record.measurement.total_s,
    )

    showcase = {
        "status": "completed",
        "rq1": rq1_summary,
        "fixed_regime": {
            "midpoint_slowdown": midpoint_slowdown,
            "selected_policy": selected_policy,
            "vanilla_total_s": fixed_vanilla_result.record.measurement.total_s,
            "hybrid_total_s": fixed_hybrid_result.record.measurement.total_s,
            "surface_equivalence": {
                "allclose": comparison["allclose"],
                "rmse": comparison["rmse"],
                "mismatch_count": comparison["mismatch_count"],
                "speedup_mean": comparison["speedup_rhs_vs_lhs_baseline"],
            },
        },
    }
    return showcase


def pipeline():
    cpu_batch_size = 4
    repeats = 1
    max_slowdown = 100.0
    jump_factor = 1.8
    linear_samples = 5
    atol = 1e-6
    rtol = 1e-5
    cpu_workers = max(1, os.cpu_count() or 1)
    seed = 1337
    subset_size = 1024
    batch_size = 64
    grid_resolution = 8
    grid_scale = 1.0
    output_root = "outputs"

    base_task = replace(
        WORKLOADS["cifar10_resnet20_classification"].spec,
        checkpoint_path="assets/cifar10-resnet20-0.pkl",
    )
    vanilla_config = ExperimentConfig(
        "pipeline",
        seed,
        "vanilla",
        base_task,
        DataConfig(
            subset_size,
            batch_size,
            None,
            64,
            0,
        ),
        GridConfig(grid_resolution, grid_scale),
        RuntimeConfig("auto", None, False, 1.0, None, output_root, False),
        ResourcesConfig(0),
    )

    hybrid_config = vanilla_config.clone()
    hybrid_config.backend = "hybrid"
    hybrid_config.data.cpu_batch_size = cpu_batch_size
    hybrid_config.resources.cpu_workers = cpu_workers

    task_specs: list[MLTaskSpec] = [
        base_task,
        replace(
            WORKLOADS["california_mlp_regression"].spec,
            checkpoint_path=None,
        ),
    ]
    workload_showcases = {}

    for task_spec in task_specs:
        workload_showcases[task_spec.name] = run_experiments(
            task_spec,
            vanilla_config=vanilla_config,
            hybrid_config=hybrid_config,
            repeats=repeats,
            max_slowdown=max_slowdown,
            jump_factor=jump_factor,
            linear_samples=linear_samples,
            atol=atol,
            rtol=rtol,
        )

    retry = 1
    model_variants = [
        "assets/cifar10-resnet20-0.pkl",
        "assets/cifar10-resnet20-123.pkl",
        "assets/cifar10-resnet20-2023.pkl",
        "assets/cifar10-resnet20-123456.pkl",
    ]
    experiment_2b_summary = experiment_2b.main(
        workload_showcases["cifar10_resnet20_classification"]["fixed_regime"][
            "midpoint_slowdown"
        ],
        retry,
        model_variants,
        base_workload=replace(
            WORKLOADS["cifar10_resnet20_classification"].spec,
            checkpoint_path="assets/cifar10-resnet20-0.pkl",
        ),
        base_runtime=hybrid_config,
    )

    summary = {
        "workload_showcases": workload_showcases,
        "experiment_2b": experiment_2b_summary,
    }
    return summary


if __name__ == "__main__":
    print_json(pipeline())
