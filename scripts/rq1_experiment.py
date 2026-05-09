#!/usr/bin/env python3
from __future__ import annotations

import statistics

from src.backends import run_backend
from src.compare import compare_surfaces
from src.system_schema import HybridMode, SchedulerRequest


def main(
    *,
    hybrid_request: SchedulerRequest,
    vanilla_surface: list[tuple[int, int, float]],
    vanilla_total_s: float,
    bracket_repeats: int,
    sample_repeats: int,
    max_slowdown: float,
    jump_factor: float,
    linear_samples: int,
    atol: float,
    rtol: float,
    seed: int,
    cpu_workers: int,
    cpu_batch_size: int,
):
    assert isinstance(hybrid_request.mode, HybridMode)

    def evaluate_slowdown(slowdown: float, repeats: int):
        runs = []

        for repeat_index in range(repeats):
            repeated_request = SchedulerRequest(
                hybrid_request.task,
                hybrid_request.grid,
                HybridMode(
                    hybrid_request.mode.gpu_batch_size,
                    cpu_batch_size,
                    cpu_workers,
                ),
            )
            runs.append(
                run_backend(
                    repeated_request,
                    seed=seed + repeat_index,
                    gpu_slowdown_factor=slowdown,
                )
            )

        hybrid_total_s_mean = statistics.fmean(
            run.record.measurement.total_s for run in runs
        )
        speedup_mean = (vanilla_total_s * slowdown) / hybrid_total_s_mean
        return {
            "slowdown": slowdown,
            "speedup_mean": speedup_mean,
            "runs": runs,
        }

    low = evaluate_slowdown(1.0, bracket_repeats)
    assert low["speedup_mean"] <= 1.0, (
        "RQ1 assumes slowdown=1.0 is not already in the positive-speedup regime"
    )

    current = 1.0
    high = None

    while current < max_slowdown:
        current = min(max_slowdown, max(current * jump_factor, current + 0.1))
        candidate = evaluate_slowdown(current, bracket_repeats)
        if candidate["speedup_mean"] > 1.0:
            high = candidate
            break
        low = candidate

    if high is None:
        raise ValueError(
            "failed to find a positive speedup under the given max_slowdown"
        )

    for _ in range(2):
        interval = high["slowdown"] - low["slowdown"]
        if interval <= 0.1:
            break
        candidate = evaluate_slowdown(
            (high["slowdown"] + low["slowdown"]) / 2.0,
            bracket_repeats,
        )
        if candidate["speedup_mean"] > 1.0:
            high = candidate
        else:
            low = candidate

    crossover_region = (low["slowdown"], high["slowdown"])
    step = (high["slowdown"] - low["slowdown"]) / float(linear_samples - 1)
    sampled_slowdowns = [
        low["slowdown"] + (index * step) for index in range(linear_samples)
    ]

    evaluations = []
    for slowdown in sampled_slowdowns:
        evaluation = evaluate_slowdown(slowdown, sample_repeats)
        repeat_metrics = []

        for run in evaluation["runs"]:
            if run.records is None:
                raise ValueError(
                    "run does not contain in-memory loss records: "
                    f"{run.record.experiment_name}"
                )
            comparison = compare_surfaces(
                lhs_surface=vanilla_surface,
                rhs_surface=run.records,
                atol=atol,
                rtol=rtol,
                lhs_total_s=vanilla_total_s * slowdown,
                rhs_total_s=run.record.measurement.total_s,
            )
            repeat_metrics.append(
                {
                    "hybrid_total_s": run.record.measurement.total_s,
                    "vanilla_total_s": vanilla_total_s * slowdown,
                    "speedup": comparison["speedup_rhs_vs_lhs_baseline"],
                    "allclose": comparison["allclose"],
                    "rmse": comparison["rmse"],
                }
            )

        evaluations.append(
            {
                "slowdown": slowdown,
                "hybrid_wins": statistics.fmean(
                    metric["speedup"] for metric in repeat_metrics
                )
                > 1.0,
                "surface_valid": all(metric["allclose"] for metric in repeat_metrics),
                "hybrid_total_s_mean": statistics.fmean(
                    metric["hybrid_total_s"] for metric in repeat_metrics
                ),
                "vanilla_total_s_mean": statistics.fmean(
                    metric["vanilla_total_s"] for metric in repeat_metrics
                ),
                "speedup_mean": statistics.fmean(
                    metric["speedup"] for metric in repeat_metrics
                ),
                "speedup_std": (
                    statistics.stdev(metric["speedup"] for metric in repeat_metrics)
                    if len(repeat_metrics) > 1
                    else None
                ),
                "rmse_mean": statistics.fmean(
                    metric["rmse"] for metric in repeat_metrics
                ),
            }
        )

    return {
        "crossover_region": crossover_region,
        "samples": evaluations,
    }
