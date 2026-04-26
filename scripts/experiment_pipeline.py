#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import statistics

from src.cli import print_json
from src.config import load_config
from src.results import load_cached_run_with_surface, load_surface
from src.results import write_json, write_summary_json
from src.runner import run_baseline_and_persist
from scripts import experiment_2b, rq1_experiment


def main():
    hybrid_config_path = "configs/hybrid.yaml"
    vanilla_config_path = "configs/vanilla.yaml"
    cpu_batch_size = 4
    repeats = 1
    max_slowdown = 100.0
    jump_factor = 1.8
    linear_samples = 5
    atol = 1e-6
    rtol = 1e-5
    cpu_workers = max(1, os.cpu_count() or 1)

    hybrid_workload = load_config(hybrid_config_path)
    hybrid_workload.resources.cpu_workers = cpu_workers
    hybrid_workload.data.cpu_batch_size = cpu_batch_size

    vanilla_workload = load_config(vanilla_config_path)
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
    gpu_slowdown_factor = statistics.fmean(rq1_summary["crossover_region"])
    write_summary_json(hybrid_workload, "results.json", rq1_summary)

    retry = 1
    model_variants = [
        "assets/cifar10-resnet20-0.pkl",
        "assets/cifar10-resnet20-123.pkl",
        "assets/cifar10-resnet20-2023.pkl",
        "assets/cifar10-resnet20-123456.pkl",
    ]

    experiment_2b_summary = experiment_2b.main(
        gpu_slowdown_factor,
        retry,
        model_variants,
        hybrid_config_path=hybrid_config_path,
    )

    summary = {
        "rq1": rq1_summary,
        "experiment_2b": experiment_2b_summary,
    }

    output_path = Path("outputs") / "experiment-pipeline-summary.json"
    write_json(output_path, summary)
    print_json(summary)


if __name__ == "__main__":
    main()
