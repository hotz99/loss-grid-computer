from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.functional_eval.experiment import (
    FunctionalEvalConfig,
    build_default_request,
    run_experiment,
)


@dataclass(frozen=True)
class PlatformSuiteScenario:
    name: str
    sample_count: int
    repeats: int = 3
    batch_size: int = 32
    resolution: int = 8
    scale: float = 1.0
    point_chunk_sizes: tuple[int, ...] = (32, 64)
    max_memory_fraction: float | None = 0.85


DEFAULT_SCENARIOS: tuple[PlatformSuiteScenario, ...] = (
    PlatformSuiteScenario(
        name="functional_seq_1024_stability",
        sample_count=1024,
        repeats=7,
        point_chunk_sizes=(),
    ),
    PlatformSuiteScenario(
        name="functional_seq_2k_stability",
        sample_count=2048,
        repeats=5,
        point_chunk_sizes=(),
    ),
)

PRD_VMAP_REPRODUCTION_SCENARIO = PlatformSuiteScenario(
    name="prd_confirmation",
    sample_count=1024,
    point_chunk_sizes=(1, 2, 4, 8, 16, 32, 64),
)

FULL_TEST_SET_SCENARIO = PlatformSuiteScenario(
    name="full_test_set_scaling",
    sample_count=0,
    point_chunk_sizes=(32, 64),
)

LEGACY_DEFAULT_SCENARIOS: tuple[PlatformSuiteScenario, ...] = (
    PRD_VMAP_REPRODUCTION_SCENARIO,
    *DEFAULT_SCENARIOS,
    FULL_TEST_SET_SCENARIO,
)

T4SuiteScenario = PlatformSuiteScenario
DEFAULT_T4_SCENARIOS = DEFAULT_SCENARIOS
PRD_CONFIRMATION_T4_SCENARIO = PRD_VMAP_REPRODUCTION_SCENARIO
FULL_TEST_SET_T4_SCENARIO = FULL_TEST_SET_SCENARIO
LEGACY_DEFAULT_T4_SCENARIOS = LEGACY_DEFAULT_SCENARIOS


def run_platform_suite(
    *,
    scenarios: Iterable[PlatformSuiteScenario] = DEFAULT_SCENARIOS,
    device: str = "cuda",
    seed: int = 1337,
    output_dir: str | Path = Path("outputs") / "functional_eval" / "platform_suite",
    run_label: str | None = None,
) -> dict[str, Any]:
    scenario_results = []
    for scenario in scenarios:
        request = build_default_request(
            device=device,
            sample_count=scenario.sample_count,
            batch_size=scenario.batch_size,
            resolution=scenario.resolution,
            scale=scenario.scale,
        )
        summary = run_experiment(
            FunctionalEvalConfig(
                request=request,
                seed=seed,
                repeats=scenario.repeats,
                point_chunk_sizes=scenario.point_chunk_sizes,
                max_memory_fraction=scenario.max_memory_fraction,
                run_label=run_label,
            )
        )
        scenario_results.append(
            {
                "scenario": asdict(scenario),
                "summary_path": summary["output_path"],
                "platform": summary["platform"],
                "config": summary["config"],
                "candidate_summary": summary["candidate_summary"],
                "paired_candidate_metrics": _paired_candidate_metrics(summary),
                "best_valid_candidate": _best_valid_candidate(summary),
                "acceptance": _acceptance(summary),
            }
        )

    suite = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "seed": seed,
        "run_label": run_label,
        "scenario_count": len(scenario_results),
        "scenarios": scenario_results,
    }
    output_path = _write_suite_summary(suite, output_dir, run_label)
    suite["output_path"] = str(output_path)
    _print_suite_table(suite)
    return suite


def run_t4_suite(**kwargs: Any) -> dict[str, Any]:
    return run_platform_suite(**kwargs)


def default_scenarios() -> tuple[PlatformSuiteScenario, ...]:
    return DEFAULT_SCENARIOS


def _best_valid_candidate(summary: dict[str, Any]) -> dict[str, Any] | None:
    candidates = _paired_candidate_metrics(summary)
    if not candidates:
        return None
    valid = [
        row
        for row in candidates
        if row["all_validations_passed"] is True
        and row["paired_speedup_mean"] is not None
    ]
    if not valid:
        return None
    return max(valid, key=lambda row: float(row["paired_speedup_mean"]))


def _acceptance(summary: dict[str, Any], threshold: float = 1.05) -> dict[str, Any]:
    best = _best_valid_candidate(summary)
    speedup = None if best is None else best["paired_speedup_mean"]
    min_speedup = None if best is None else best["paired_speedup_min"]
    return {
        "threshold_speedup": threshold,
        "met": (
            speedup is not None
            and min_speedup is not None
            and float(speedup) >= threshold
            and float(min_speedup) >= 1.0
        ),
        "candidate": None if best is None else best["candidate"],
        "paired_speedup_mean": speedup,
        "paired_speedup_min": min_speedup,
        "all_repeats_beat_baseline": (
            None if best is None else best["all_repeats_beat_baseline"]
        ),
    }


def _paired_candidate_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    runs = summary.get("runs", [])
    baselines = {
        int(run["repeat"]): run
        for run in runs
        if run["candidate"] == "baseline_original"
        and run["status"] == "ok"
        and run["total_grid_s"] is not None
    }
    candidates = [
        row for row in summary["candidate_summary"]
        if row["candidate"] != "baseline_original"
    ]

    metrics: list[dict[str, Any]] = []
    for row in candidates:
        candidate_runs = [
            run
            for run in runs
            if run["candidate"] == row["candidate"]
            and run["status"] == "ok"
            and run["total_grid_s"] is not None
        ]
        paired_speedups = []
        validations = []
        for run in candidate_runs:
            baseline = baselines.get(int(run["repeat"]))
            if baseline is not None:
                paired_speedups.append(
                    float(baseline["total_grid_s"]) / float(run["total_grid_s"])
                )
            if run.get("validation") is not None:
                validations.append(bool(run["validation"].get("allclose")))

        metrics.append(
            {
                "candidate": row["candidate"],
                "taxonomy": row.get("taxonomy"),
                "status_counts": row["status_counts"],
                "mean_total_grid_s": row.get("mean_total_grid_s"),
                "summary_mean_speedup_vs_baseline": row.get(
                    "mean_speedup_vs_baseline"
                ),
                "all_validations_passed": all(validations) if validations else None,
                "paired_speedups": paired_speedups,
                "paired_speedup_mean": _mean(paired_speedups),
                "paired_speedup_min": min(paired_speedups) if paired_speedups else None,
                "paired_speedup_max": max(paired_speedups) if paired_speedups else None,
                "paired_speedup_stdev": _stdev(paired_speedups),
                "all_repeats_beat_baseline": (
                    all(speedup >= 1.0 for speedup in paired_speedups)
                    if paired_speedups else None
                ),
            }
        )
    return metrics


def _write_suite_summary(
    suite: dict[str, Any],
    output_dir: str | Path,
    run_label: str | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = output_path / f"{_filename_label(run_label)}{timestamp}-platform-suite.json"
    suite["output_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(_json_safe(suite), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _print_suite_table(suite: dict[str, Any]) -> None:
    print(f"functional_eval platform suite summary={suite['output_path']}")
    print("scenario                 best_candidate            speedup   accepted")
    for item in suite["scenarios"]:
        acceptance = item["acceptance"]
        print(
            f"{item['scenario']['name']:<24} "
            f"{str(acceptance['candidate']):<25.25} "
            f"{_format_float(acceptance['paired_speedup_mean']):>8} "
            f"{str(acceptance['met']):>10}"
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _format_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _filename_label(value: str | None) -> str:
    if value is None or not value.strip():
        return ""
    safe = "".join(
        char if char.isalnum() or char in ("-", "_") else "-"
        for char in value.strip().lower()
    ).strip("-_")
    return f"{safe}-" if safe else ""


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _parse_chunks(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in raw.split(",") if value.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the platform-agnostic functional evaluation suite.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--sample-counts",
        default="1024,2048",
        help="Comma-separated sample counts. Use 0 for the full CIFAR-10 test set.",
    )
    parser.add_argument(
        "--point-chunk-sizes",
        default="",
        help=(
            "Comma-separated vmapped point chunk sizes for all custom scenarios. "
            "Leave empty to run baseline plus functional_sequential only."
        ),
    )
    parser.add_argument(
        "--include-prd-confirmation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the original full chunk sweep used to reproduce the vmap result.",
    )
    parser.add_argument(
        "--include-full-test-set",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the full CIFAR-10 test-set scaling scenario.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("outputs") / "functional_eval" / "platform_suite"),
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional filename prefix for separating runs from different machines.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios: list[PlatformSuiteScenario] = []
    if args.include_prd_confirmation:
        scenarios.append(PRD_VMAP_REPRODUCTION_SCENARIO)

    chunks = _parse_chunks(args.point_chunk_sizes)
    for sample_count in (int(value) for value in args.sample_counts.split(",") if value.strip()):
        if args.include_prd_confirmation and sample_count == 1024:
            continue
        if sample_count == 0 and not args.include_full_test_set:
            continue
        name = "full_test_set_scaling" if sample_count == 0 else f"focused_{sample_count}_scaling"
        scenarios.append(
            PlatformSuiteScenario(
                name=name,
                sample_count=sample_count,
                repeats=args.repeats,
                batch_size=args.batch_size,
                resolution=args.resolution,
                scale=args.scale,
                point_chunk_sizes=chunks,
            )
        )

    run_platform_suite(
        scenarios=tuple(scenarios),
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        run_label=args.run_label,
    )


if __name__ == "__main__":
    main()
