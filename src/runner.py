from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from src.backends import run_backend
from src.compare import compare_run_outputs
from src.config import ExperimentConfig, experiment_config_from_dict, load_config
from src.results import ComparisonRecord, write_experiment_result

DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-5


def _set_dotted(raw: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = raw
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def expand_sweep(config: ExperimentConfig) -> list[ExperimentConfig]:
    raw = config.to_dict()
    keys = list(config.sweep.keys())
    values = [config.sweep[key] for key in keys]
    expanded = []
    for combination in itertools.product(*values):
        instance = experiment_config_from_dict(raw)
        instance_raw = instance.to_dict()
        for key, value in zip(keys, combination):
            _set_dotted(instance_raw, key, value)
        instance = experiment_config_from_dict(instance_raw)
        instance.sweep = {}
        expanded.append(instance)
    return expanded


def _find_latest_baseline_run(baseline_config: ExperimentConfig) -> Path:
    output_root = Path(baseline_config.runtime.output_root)
    prefix = f"{baseline_config.experiment_name}-"
    expected_config = baseline_config.to_dict()
    matches = [
        path
        for path in output_root.iterdir()
        if path.is_dir()
        and path.name.startswith(prefix)
        and (path / "summary.json").exists()
        and (path / "config.snapshot.json").exists()
        and _load_json(path / "config.snapshot.json") == expected_config
    ]
    if not matches:
        raise FileNotFoundError(
            "No cached baseline run matches the current baseline config under "
            f"{output_root} with prefix {prefix!r}"
        )
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_existing_result(run_dir: Path):
    summary = _load_json(run_dir / "summary.json")
    runtime_log = _load_json(run_dir / "runtime_breakdown.json")
    return {
        "output_dir": str(run_dir),
        "summary": summary,
        "runtime_log": runtime_log,
    }


def resolve_baseline(baseline_config_path: str, cached: bool):
    baseline_config = load_config(baseline_config_path)
    if cached:
        return _load_existing_result(_find_latest_baseline_run(baseline_config))
    baseline_result = run_backend(baseline_config)
    write_experiment_result(baseline_result)
    return {
        "output_dir": baseline_result.record.output_dir,
        "summary": baseline_result.record,
        "runtime_log": baseline_result.runtime_log,
    }


def _execution_total_s(summary) -> float:
    if isinstance(summary, dict):
        return float(summary["measurement"]["total_s"])
    return float(summary.measurement.total_s)


def _comparison_record(baseline_result, result, comparison) -> ComparisonRecord:
    baseline_total_s = _execution_total_s(baseline_result["summary"])
    result_total_s = _execution_total_s(result.record)
    speedup = None
    if result_total_s > 0:
        speedup = baseline_total_s / result_total_s
    return ComparisonRecord(
        baseline_total_s=baseline_total_s,
        speedup=speedup,
        allclose=comparison.get("allclose"),
        rmse=comparison.get("rmse"),
    )


def _print_runtime_diagnostics(result) -> None:
    scheduler = result.runtime_log.get("hybrid_scheduler")
    if not scheduler:
        return

    cpu = scheduler.get("cpu", {})
    gpu = scheduler.get("gpu", {})

    cpu_max_wall = cpu.get("cpu_max_wall_time_s")
    cpu_points = cpu.get("total_points_processed")
    per_worker = cpu.get("points_processed")
    cpu_max_wall_text = (
        f"{cpu_max_wall:.6f}" if isinstance(cpu_max_wall, (int, float)) else "n/a"
    )

    gpu_wall = gpu.get("wall_time_s")
    gpu_preload = gpu.get("preload_s")
    gpu_compile = gpu.get("compile_s")
    gpu_wall_text = f"{gpu_wall:.6f}" if isinstance(gpu_wall, (int, float)) else "n/a"
    gpu_preload_text = (
        f"{gpu_preload:.6f}" if isinstance(gpu_preload, (int, float)) else "n/a"
    )
    gpu_compile_text = (
        f"{gpu_compile:.6f}" if isinstance(gpu_compile, (int, float)) else "n/a"
    )
    print(
        "[hybrid] "
        f"gpu_preload={gpu_preload_text} "
        f"gpu_compile={gpu_compile_text} "
        f"gpu_wall={gpu_wall_text} "
        f"cpu_max_wall={cpu_max_wall_text} "
        f"cpu_points={cpu_points} "
        f"per_worker={per_worker}"
    )


def _printable_record(result) -> dict[str, Any]:
    record = asdict(result.record)
    record.pop("config", None)
    return record


def _print_baseline_summary(baseline_result) -> None:
    summary = baseline_result["summary"]
    if isinstance(summary, dict):
        record = dict(summary)
    else:
        record = asdict(summary)
    record.pop("config", None)
    print("[baseline] resolved")
    print(json.dumps(record, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loss-grid experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument("--config", required=True, help="YAML or JSON config")
    run_parser.add_argument("--bench", help="Baseline config to run or reuse first")
    run_parser.add_argument(
        "--cached",
        action="store_true",
        help="Reuse the latest run for the baseline config",
    )

    sweep_parser = subparsers.add_parser("sweep", help="Run a config sweep")
    sweep_parser.add_argument(
        "--config", required=True, help="YAML or JSON sweep config"
    )
    sweep_parser.add_argument("--bench", help="Baseline config to run or reuse first")
    sweep_parser.add_argument(
        "--cached",
        action="store_true",
        help="Reuse the latest run for the baseline config",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two run outputs numerically"
    )
    compare_parser.add_argument(
        "--lhs", required=True, help="Baseline run directory or loss_surface.pt"
    )
    compare_parser.add_argument(
        "--rhs", required=True, help="Candidate run directory or loss_surface.pt"
    )
    compare_parser.add_argument(
        "--atol", type=float, default=1e-6, help="Absolute tolerance"
    )
    compare_parser.add_argument(
        "--rtol", type=float, default=1e-5, help="Relative tolerance"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compare":
        comparison = compare_run_outputs(
            lhs_path=args.lhs, rhs_path=args.rhs, atol=args.atol, rtol=args.rtol
        )
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return

    config = load_config(args.config)
    baseline_result = (
        resolve_baseline(args.bench, args.cached)
        if getattr(args, "bench", None)
        else None
    )
    if baseline_result is not None:
        _print_baseline_summary(baseline_result)

    if args.command == "run":
        result = run_backend(config)
        write_experiment_result(result)
        if baseline_result is not None:
            comparison = compare_run_outputs(
                lhs_path=baseline_result["output_dir"],
                rhs_path=result.record.output_dir,
                atol=DEFAULT_ATOL,
                rtol=DEFAULT_RTOL,
            )
            result.record = replace(
                result.record,
                comparison=_comparison_record(baseline_result, result, comparison),
            )
            write_experiment_result(result)
        print(json.dumps(_printable_record(result), indent=2, sort_keys=True))
        _print_runtime_diagnostics(result)
        return

    if args.command == "sweep":
        results = []
        for index, expanded in enumerate(expand_sweep(config)):
            expanded.experiment_name = f"{expanded.experiment_name}-run{index:03d}"
            result = run_backend(expanded)
            write_experiment_result(result)
            if baseline_result is not None:
                comparison = compare_run_outputs(
                    lhs_path=baseline_result["output_dir"],
                    rhs_path=result.record.output_dir,
                    atol=DEFAULT_ATOL,
                    rtol=DEFAULT_RTOL,
                )
                result.record = replace(
                    result.record,
                    comparison=_comparison_record(baseline_result, result, comparison),
                )
                write_experiment_result(result)
            results.append(result)

        summary_records = []
        total_configs = len(results)
        for index, result in enumerate(results):
            summary_records.append(asdict(result.record))
            print(
                f"[{index + 1}/{total_configs}] completed {result.record.experiment_name}"
            )
            print(json.dumps(_printable_record(result), indent=2, sort_keys=True))
            _print_runtime_diagnostics(result)

        if summary_records:
            summary_dir = (
                Path(config.runtime.output_root)
                / f"{config.experiment_name}-sweep-summary"
            )
            summary_dir.mkdir(parents=True, exist_ok=True)
            csv_path = summary_dir / "results.csv"
            fieldnames = list(summary_records[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_records)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
