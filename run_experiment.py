#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from loss_grid import load_config, make_executor
from loss_grid.compare import compare_run_outputs
from loss_grid.profiling import enable_profiling, get_profiler
from loss_grid.results import write_experiment_result, write_summary_table
from loss_grid.sweep import expand_sweep_configs


def _print_runtime_diagnostics(result) -> None:
    scheduler = result.runtime_log.get("hybrid_scheduler")
    if scheduler:
        gpu_wall = scheduler.get("gpu_worker_wall_s_after_calibration")
        cpu_max_wall = scheduler.get("cpu_worker_wall_s_max_after_calibration")
        cpu_points = scheduler.get("cpu_points_processed_after_calibration")
        per_worker = scheduler.get("cpu_worker_points_processed_after_calibration")
        gpu_wall_text = f"{gpu_wall:.6f}" if isinstance(gpu_wall, (int, float)) else "n/a"
        cpu_max_wall_text = f"{cpu_max_wall:.6f}" if isinstance(cpu_max_wall, (int, float)) else "n/a"
        print(
            "[hybrid] "
            f"gpu_wall={gpu_wall_text} "
            f"cpu_max_wall={cpu_max_wall_text} "
            f"cpu_points={cpu_points} "
            f"per_worker={per_worker}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loss-grid experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument("--config", required=True, help="YAML or JSON config")

    sweep_parser = subparsers.add_parser("sweep", help="Run a config sweep")
    sweep_parser.add_argument("--config", required=True, help="YAML or JSON sweep config")

    compare_parser = subparsers.add_parser("compare", help="Compare two run outputs numerically")
    compare_parser.add_argument("--lhs", required=True, help="Baseline run directory or loss_surface.pt")
    compare_parser.add_argument("--rhs", required=True, help="Candidate run directory or loss_surface.pt")
    compare_parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    compare_parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")

    return parser


def run_single(config_path: str) -> None:
    enable_profiling()
    profiler = get_profiler()
    
    config = load_config(config_path)
    executor = make_executor(config)
    
    profiler.snapshot("experiment_start")
    result = executor.run(config)
    profiler.snapshot("experiment_complete")
    
    if result.is_root:
        # Add profiling data to result
        result.runtime_log["profiling"] = profiler.summarize()
        
        write_experiment_result(result)
        print(json.dumps(result.summary_record(), indent=2, sort_keys=True))
        _print_runtime_diagnostics(result)
        
        # Print profiling summary
        profiler.print_summary()


def run_sweep(config_path: str) -> None:
    enable_profiling()
    
    root_config = load_config(config_path)
    base_name = root_config.experiment_name
    configs = expand_sweep_configs(root_config)
    summary_records = []
    total_configs = len(configs)
    for index, config in enumerate(configs):
        profiler = get_profiler()
        profiler.snapshots.clear()
        profiler.sections.clear()
        profiler.active_sections.clear()
        
        config.experiment_name = f"{config.experiment_name}-run{index:03d}"
        executor = make_executor(config)
        
        profiler.snapshot("experiment_start")
        result = executor.run(config)
        profiler.snapshot("experiment_complete")
        
        if result.is_root:
            result.runtime_log["profiling"] = profiler.summarize()
            write_experiment_result(result)
            record = result.summary_record()
            summary_records.append(record)
            print(f"[{index + 1}/{total_configs}] completed {config.experiment_name}")
            print(json.dumps(record, indent=2, sort_keys=True))
            _print_runtime_diagnostics(result)
            profiler.print_summary()
    if summary_records:
        summary_dir = Path(root_config.runtime.output_root) / f"{base_name}-sweep-summary"
        write_summary_table(
            summary_records,
            output_dir=str(summary_dir),
            output_formats=root_config.runtime.output_formats,
        )


def compare_outputs(lhs: str, rhs: str, atol: float, rtol: float) -> None:
    comparison = compare_run_outputs(lhs_path=lhs, rhs_path=rhs, atol=atol, rtol=rtol)
    print(json.dumps(comparison, indent=2, sort_keys=True))


def main() -> None:
    time.sleep(5)
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_single(args.config)
        return
    if args.command == "sweep":
        run_sweep(args.config)
        return
    if args.command == "compare":
        compare_outputs(args.lhs, args.rhs, args.atol, args.rtol)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
