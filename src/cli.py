from __future__ import annotations

import argparse
from pathlib import Path
import runpy

from src.calibration import run_calibration
from src.config import (
    load_config,
)
from src.results import (
    load_cached_run_summary,
    to_pretty_json,
    write_summary_json,
)
from src.runner import run_baseline_and_persist


def print_json(value: object) -> None:
    print(to_pretty_json(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loss-grid experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "rq1",
        help="Run the configured RQ1 experiment script",
    )

    subparsers.add_parser(
        "2b",
        help="Run the configured Experiment 2B cache-amortization script",
    )

    calibration_parser = subparsers.add_parser(
        "calibrate",
        help="Run the calibration search over CPU workers and CPU batch sizes",
    )
    calibration_parser.add_argument(
        "--config",
        required=True,
        help="Hybrid config with fixed slowdown and workload settings",
    )
    calibration_parser.add_argument(
        "--bench",
        required=True,
        help="Baseline config path",
    )
    calibration_parser.add_argument(
        "--cached",
        action="store_true",
        help="Reuse the latest run for the baseline config",
    )
    calibration_parser.add_argument(
        "--cpu-workers",
        required=True,
        help="Comma-separated CPU worker counts, for example 1,2,4,8",
    )
    calibration_parser.add_argument(
        "--cpu-batch-sizes",
        required=True,
        help="Comma-separated CPU batch sizes, for example 4,8,16",
    )
    calibration_parser.add_argument(
        "--retry",
        type=int,
        default=1,
        help="Stop after this many consecutive non-improvements over CPU worker counts",
    )
    calibration_parser.add_argument(
        "--slowdown",
        type=float,
        help="Override config.runtime.gpu_slowdown_factor before running calibration",
    )
    return parser


def _parse_int_tuple(raw: str, flag: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"{flag} must contain at least one integer")
    return values

def main():
    args = build_parser().parse_args()

    if args.command == "rq1":
        runpy.run_path(
            str(Path(__file__).resolve().parent.parent / "scripts" / "rq1_experiment.py"),
            run_name="__main__",
        )
        return

    if args.command == "2b":
        runpy.run_path(
            str(Path(__file__).resolve().parent.parent / "scripts" / "experiment_2b.py"),
            run_name="__main__",
        )
        return

    if args.command == "calibrate":
        config = load_config(args.config)
        baseline_config = load_config(args.bench)
        if args.retry < 1:
            raise ValueError("--retry must be >= 1")
        if args.slowdown is not None:
            if args.slowdown < 1.0:
                raise ValueError("--slowdown must be >= 1.0")
            config.runtime.gpu_slowdown_factor = args.slowdown
            baseline_config.runtime.gpu_slowdown_factor = args.slowdown
        cpu_worker_values = _parse_int_tuple(args.cpu_workers, "--cpu-workers")
        cpu_batch_sizes = _parse_int_tuple(args.cpu_batch_sizes, "--cpu-batch-sizes")
        if any(value < 1 for value in cpu_worker_values):
            raise ValueError("--cpu-workers values must be >= 1")
        if any(value < 1 for value in cpu_batch_sizes):
            raise ValueError("--cpu-batch-sizes values must be >= 1")

        try:
            baseline_summary = load_cached_run_summary(baseline_config)
        except FileNotFoundError:
            baseline_summary = run_baseline_and_persist(baseline_config).record
        results = run_calibration(
            config,
            baseline_summary.measurement.total_s,
            cpu_worker_values,
            cpu_batch_sizes,
            args.retry,
        )
        write_summary_json(config, "calibration.json", results)
        print_json(results)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
