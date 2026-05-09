#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, replace
import statistics

from src.backends import run_backend
from src.calibration import (
    run_calibration,
    resolve_available_cpu_cores,
    resolve_cpu_batch_size_candidates,
    resolve_cpu_worker_candidates,
)
from src.compare import compare_surfaces
from src.results import print_json
from src.system_schema import GridSpec, HybridMode, MLTaskSpec, SchedulerRequest, VanillaMode
from src.workloads import WORKLOADS
from scripts import experiment_2b, rq1_experiment


def run_experiments(
    task_spec: MLTaskSpec,
    grid: GridSpec,
    gpu_batch_size: int,
    cpu_batch_size: int,
    cpu_workers: int,
    seed: int,
    bracket_repeats: int,
    sample_repeats: int,
    max_slowdown: float,
    jump_factor: float,
    linear_samples: int,
    atol: float,
    rtol: float,
):
    vanilla_request = SchedulerRequest(task_spec, grid, VanillaMode(gpu_batch_size))
    hybrid_request = SchedulerRequest(
        task_spec,
        grid,
        HybridMode(gpu_batch_size, cpu_batch_size, cpu_workers),
    )
    vanilla_result = run_backend(vanilla_request, seed)

    rq1_summary = rq1_experiment.main(
        hybrid_request=hybrid_request,
        vanilla_surface=vanilla_result.records,
        vanilla_total_s=vanilla_result.record.measurement.total_s,
        bracket_repeats=bracket_repeats,
        sample_repeats=sample_repeats,
        max_slowdown=max_slowdown,
        jump_factor=jump_factor,
        linear_samples=linear_samples,
        atol=atol,
        rtol=rtol,
        seed=seed,
        cpu_workers=cpu_workers,
        cpu_batch_size=cpu_batch_size,
    )

    midpoint_slowdown = statistics.fmean(rq1_summary["crossover_region"])
    cpu_worker_values = resolve_cpu_worker_candidates()
    cpu_batch_sizes = resolve_cpu_batch_size_candidates(
        task_spec.dataset.sample_count,
        gpu_batch_size,
    )

    # calibration: select best policy at the midpoint throughput regime
    selected_mode = run_calibration(
        SchedulerRequest(task_spec, grid, HybridMode(gpu_batch_size)),
        vanilla_result.record.measurement.total_s * midpoint_slowdown,
        cpu_worker_values,
        cpu_batch_sizes,
        1,
        seed,
        midpoint_slowdown,
    )

    # static hybrid baseline: median worker count, fixed batch size
    # represents a practitioner's reasonable default without calibration
    static_workers = cpu_worker_values[len(cpu_worker_values) // 2]
    static_batch_size = min(8, cpu_batch_sizes[-1])
    static_mode = HybridMode(gpu_batch_size, static_batch_size, static_workers)

    fixed_vanilla_result = run_backend(vanilla_request, seed, midpoint_slowdown)
    fixed_hybrid_result = run_backend(
        SchedulerRequest(task_spec, grid, selected_mode),
        seed,
        midpoint_slowdown,
    )
    fixed_static_result = run_backend(
        SchedulerRequest(task_spec, grid, static_mode),
        seed,
        midpoint_slowdown,
    )

    hybrid_comparison = compare_surfaces(
        fixed_vanilla_result.records,
        fixed_hybrid_result.records,
        atol,
        rtol,
        fixed_vanilla_result.record.measurement.total_s,
        fixed_hybrid_result.record.measurement.total_s,
    )
    static_comparison = compare_surfaces(
        fixed_vanilla_result.records,
        fixed_static_result.records,
        atol,
        rtol,
        fixed_vanilla_result.record.measurement.total_s,
        fixed_static_result.record.measurement.total_s,
    )

    return {
        "rq1": rq1_summary,
        "fixed_regime": {
            "midpoint_slowdown": midpoint_slowdown,
            "vanilla_total_s": fixed_vanilla_result.record.measurement.total_s,
            "calibrated_hybrid": {
                "selected_mode": asdict(selected_mode),
                "total_s": fixed_hybrid_result.record.measurement.total_s,
                "surface_equivalence": {
                    "allclose": hybrid_comparison["allclose"],
                    "rmse": hybrid_comparison["rmse"],
                    "mismatch_count": hybrid_comparison["mismatch_count"],
                    "speedup_vs_vanilla": hybrid_comparison["speedup_rhs_vs_lhs_baseline"],
                },
            },
            "static_hybrid_baseline": {
                "config": {
                    "cpu_workers": static_workers,
                    "cpu_batch_size": static_batch_size,
                },
                "total_s": fixed_static_result.record.measurement.total_s,
                "surface_equivalence": {
                    "allclose": static_comparison["allclose"],
                    "rmse": static_comparison["rmse"],
                    "mismatch_count": static_comparison["mismatch_count"],
                    "speedup_vs_vanilla": static_comparison["speedup_rhs_vs_lhs_baseline"],
                },
            },
        },
    }


def pipeline(
    seed: int = 1337,
    gpu_batch_size: int = 64,
    grid_resolution: int = 8,
    bracket_repeats: int = 1,
    sample_repeats: int = 3,
    max_slowdown: float = 100.0,
    jump_factor: float = 1.8,
    linear_samples: int = 5,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    measure_without_cache: bool = False,
):
    cpu_batch_size = 4
    cpu_workers = resolve_available_cpu_cores()
    grid = GridSpec(grid_resolution, 1.0)

    base_task = replace(
        WORKLOADS["cifar10_resnet20_classification"].spec,
        checkpoint_path="assets/cifar10-resnet20-0.pkl",
    )
    task_specs = [
        base_task,
        replace(WORKLOADS["california_mlp_regression"].spec, checkpoint_path=None),
    ]
    workload_showcases = {
        task_spec.name: run_experiments(
            task_spec,
            grid,
            gpu_batch_size,
            cpu_batch_size,
            cpu_workers,
            seed,
            bracket_repeats,
            sample_repeats,
            max_slowdown,
            jump_factor,
            linear_samples,
            atol,
            rtol,
        )
        for task_spec in task_specs
    }

    model_variants = [
        "assets/cifar10-resnet20-0.pkl",
        "assets/cifar10-resnet20-123.pkl",
        "assets/cifar10-resnet20-2023.pkl",
        "assets/cifar10-resnet20-123456.pkl",
    ]
    cifar_midpoint = workload_showcases["cifar10_resnet20_classification"]["fixed_regime"][
        "midpoint_slowdown"
    ]
    experiment_2b_summary = experiment_2b.main(
        cifar_midpoint,
        1,
        model_variants,
        base_workload=base_task,
        grid=grid,
        gpu_batch_size=gpu_batch_size,
        seed=seed,
        measure_without_cache=measure_without_cache,
    )

    return {
        "workload_showcases": workload_showcases,
        "experiment_2b": experiment_2b_summary,
    }


if __name__ == "__main__":
    print_json(pipeline())
