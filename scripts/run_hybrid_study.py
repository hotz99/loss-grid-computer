#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loss_grid import load_config, make_executor
from loss_grid.compare import compare_run_outputs
from loss_grid.profiling import enable_profiling, get_profiler
from loss_grid.results import write_experiment_result
from loss_grid.sweep import expand_sweep_configs


def _iso_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _reset_profiler() -> None:
    profiler = get_profiler()
    profiler.snapshots.clear()
    profiler.sections.clear()
    profiler.active_sections.clear()


def _run_config(config) -> Any:
    _reset_profiler()
    profiler = get_profiler()
    executor = make_executor(config)
    profiler.snapshot("experiment_start")
    result = executor.run(config)
    profiler.snapshot("experiment_complete")
    if result.is_root:
        result.runtime_log["profiling"] = profiler.summarize()
        write_experiment_result(result)
    return result


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_baseline_run(baseline_config) -> Path:
    output_root = Path(baseline_config.runtime.output_root)
    prefix = f"{baseline_config.experiment_name}-"
    matches = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and (path / "summary.json").exists()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No baseline runs found under {output_root} matching prefix {prefix!r}"
        )
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def _load_existing_result(run_dir: Path) -> Any:
    summary = _load_json(run_dir / "summary.json")
    runtime_log = _load_json(run_dir / "runtime_breakdown.json")
    return SimpleNamespace(
        output_dir=str(run_dir),
        runtime_log=runtime_log,
        is_root=True,
        summary_record=lambda: summary,
    )


def _scheduler_fields(runtime_log: Dict[str, Any]) -> Dict[str, Any]:
    scheduler = runtime_log.get("hybrid_scheduler", {}) or {}
    return {
        "cpu_workers": scheduler.get("cpu_workers"),
        "cpu_chunk_size": scheduler.get("cpu_chunk_size"),
        "gpu_chunk_size": scheduler.get("gpu_chunk_size"),
        "gpu_initial_ratio": scheduler.get("gpu_initial_ratio"),
        "gpu_initial_points": scheduler.get("gpu_initial_points"),
        "cpu_remaining_points": scheduler.get("cpu_remaining_points"),
        "cpu_helpers_enabled": scheduler.get("cpu_helpers_enabled"),
        "gpu_points_processed": scheduler.get("gpu_points_processed"),
        "cpu_points_processed": scheduler.get("cpu_points_processed"),
        "gpu_worker_wall_s": scheduler.get("gpu_worker_wall_s"),
        "cpu_worker_wall_s_max": scheduler.get("cpu_worker_wall_s_max"),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    baseline_result: Any,
    baseline_config_path: str,
    experiment_config_path: str,
    rows: List[Dict[str, Any]],
) -> None:
    best_equivalent = [
        row
        for row in rows
        if row.get("allclose") is True and isinstance(row.get("relative_speed_vs_baseline"), (int, float))
    ]
    best_equivalent.sort(key=lambda row: row["relative_speed_vs_baseline"], reverse=True)
    best_row = best_equivalent[0] if best_equivalent else None

    lines = [
        "# Baseline vs Hybrid Study",
        "",
        f"- baseline config: `{baseline_config_path}`",
        f"- experiment config: `{experiment_config_path}`",
        f"- baseline run dir: `{baseline_result.output_dir}`",
        f"- baseline total_s: `{baseline_result.summary_record().get('total_s')}`",
        f"- baseline throughput_points_per_s: `{baseline_result.summary_record().get('throughput_points_per_s')}`",
        "",
        "## Best Equivalent Hybrid Run",
        "",
    ]
    if best_row is None:
        lines.extend(
            [
                "No hybrid run was both numerically equivalent and faster than baseline.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- experiment_name: `{best_row['experiment_name']}`",
                f"- run_dir: `{best_row['run_dir']}`",
                f"- total_s: `{best_row['total_s']}`",
                f"- throughput_points_per_s: `{best_row['throughput_points_per_s']}`",
                f"- relative_speed_vs_baseline: `{best_row['relative_speed_vs_baseline']}`",
                f"- cpu_workers: `{best_row['cpu_workers']}`",
                f"- cpu_chunk_size: `{best_row['cpu_chunk_size']}`",
                f"- gpu_chunk_size: `{best_row['gpu_chunk_size']}`",
                f"- gpu_initial_ratio: `{best_row['gpu_initial_ratio']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Artifact Files",
            "",
            "- `baseline_summary.json`",
            "- `hybrid_results.csv`",
            "- `hybrid_results.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a baseline vs hybrid end-to-end grid-processing study")
    parser.add_argument("--baseline-config", required=True, help="Single-run baseline config")
    parser.add_argument(
        "--experiment-config",
        "--candidate-config",
        dest="experiment_config",
        required=True,
        help="Sweep config for hybrid experiment runs",
    )
    parser.add_argument(
        "--study-name",
        default="hybrid-study",
        help="Name prefix for the bundled artifact directory",
    )
    parser.add_argument(
        "--baseline-run-dir",
        help="Reuse an existing baseline run directory instead of rerunning baseline",
    )
    parser.add_argument(
        "--reuse-latest-baseline",
        action="store_true",
        help="Reuse the latest baseline run matching the baseline config experiment name",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    enable_profiling()

    baseline_config = load_config(args.baseline_config)
    if args.baseline_run_dir:
        baseline_result = _load_existing_result(Path(args.baseline_run_dir))
    elif args.reuse_latest_baseline:
        baseline_result = _load_existing_result(_find_latest_baseline_run(baseline_config))
    else:
        baseline_result = _run_config(baseline_config)
        if not baseline_result.is_root:
            raise RuntimeError("Baseline run did not produce a root result")

    experiment_root = load_config(args.experiment_config)
    experiment_configs = expand_sweep_configs(experiment_root)

    study_dir = Path(baseline_config.runtime.output_root) / f"{args.study_name}-{_iso_stamp()}"
    study_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = {
        "config_path": args.baseline_config,
        "output_dir": baseline_result.output_dir,
        "summary": baseline_result.summary_record(),
        "scheduler": baseline_result.runtime_log.get("hybrid_scheduler"),
    }
    _write_json(study_dir / "baseline_summary.json", baseline_summary)

    hybrid_rows: List[Dict[str, Any]] = []
    total_experiments = len(experiment_configs)
    for index, config in enumerate(experiment_configs):
        config.experiment_name = f"{config.experiment_name}-run{index:03d}"
        result = _run_config(config)
        if not result.is_root:
            continue
        comparison = compare_run_outputs(
            lhs_path=baseline_result.output_dir,
            rhs_path=result.output_dir,
            atol=1e-6,
            rtol=1e-5,
        )
        row = {
            "experiment_name": config.experiment_name,
            "run_dir": result.output_dir,
            "total_s": result.summary_record().get("total_s"),
            "throughput_points_per_s": result.summary_record().get(
                "throughput_points_per_s"
            ),
            "baseline_total_s": baseline_result.summary_record().get("total_s"),
            "relative_speed_vs_baseline": (
                baseline_result.summary_record().get("total_s")
                / result.summary_record().get("total_s")
                if result.summary_record().get("total_s")
                else None
            ),
            "allclose": comparison.get("allclose"),
            "nan_mismatch_count": comparison.get("nan_mismatch_count"),
            "max_abs_diff": comparison.get("max_abs_diff"),
            "mean_abs_diff": comparison.get("mean_abs_diff"),
            "rmse": comparison.get("rmse"),
            "runtime_delta_s": comparison.get("runtime_delta_s"),
            "compare_speedup_rhs_vs_lhs_baseline": comparison.get(
                "speedup_rhs_vs_lhs_baseline"
            ),
            **_scheduler_fields(result.runtime_log),
        }
        hybrid_rows.append(row)
        print(f"[{index + 1}/{total_experiments}] {config.experiment_name}")
        print(json.dumps(row, indent=2, sort_keys=True))

    _write_json(study_dir / "hybrid_results.json", hybrid_rows)
    _write_csv(study_dir / "hybrid_results.csv", hybrid_rows)
    _write_markdown(
        study_dir / "README.md",
        baseline_result=baseline_result,
        baseline_config_path=args.baseline_config,
        experiment_config_path=args.experiment_config,
        rows=hybrid_rows,
    )

    print(f"Study artifact written to: {study_dir}")


if __name__ == "__main__":
    main()
