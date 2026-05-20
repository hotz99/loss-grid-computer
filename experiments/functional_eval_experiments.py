from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from src.functional_eval.api_pipeline import run_pipeline as run_functional_api_probe
from src.functional_eval.experiment import (
    DEFAULT_WORKLOAD_NAME,
    FunctionalEvalConfig,
    build_default_request,
    run_experiment,
)
from src.workloads import WORKLOADS, WorkloadDefinition


DEFAULT_FUNCTIONAL_EVAL_WORKLOADS: tuple[str, ...] = (
    "cifar10_resnet20_classification",
    "cifar10_row_gru_classification",
    "california_mlp_regression",
    "mnist_mlp_classification",
)
DEFAULT_FUNCTIONAL_EVAL_SAMPLE_COUNTS: tuple[int, ...] = (1024, 2048)


@dataclass(frozen=True)
class PlatformSuiteScenario:
    name: str
    sample_count: int
    workload_name: str = DEFAULT_WORKLOAD_NAME
    repeats: int = 5
    warmup_repeats: int = 2
    batch_size: int = 32
    resolution: int = 8
    scale: float = 1.0
    point_chunk_sizes: tuple[int, ...] = (32, 64)
    max_memory_fraction: float | None = 0.85


DEFAULT_SCENARIOS: tuple[PlatformSuiteScenario, ...] = (
    *(
        PlatformSuiteScenario(
            name=f"{workload_name}__functional_seq_1024_stability",
            sample_count=1024,
            workload_name=workload_name,
            repeats=7,
            point_chunk_sizes=(),
        )
        for workload_name in DEFAULT_FUNCTIONAL_EVAL_WORKLOADS
    ),
    *(
        PlatformSuiteScenario(
            name=f"{workload_name}__functional_seq_2k_stability",
            sample_count=2048,
            workload_name=workload_name,
            repeats=5,
            point_chunk_sizes=(),
        )
        for workload_name in DEFAULT_FUNCTIONAL_EVAL_WORKLOADS
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

def run_platform_suite(
    *,
    scenarios: Iterable[PlatformSuiteScenario] = DEFAULT_SCENARIOS,
    device: str = "cuda",
    seed: int = 1337,
    output_dir: str | Path = Path("outputs") / "functional_eval" / "platform_suite",
    run_label: str | None = None,
) -> dict[str, Any]:
    del output_dir
    scenario_results = []
    for scenario in scenarios:
        definition = WORKLOADS.get(scenario.workload_name)
        try:
            request = build_default_request(
                workload_name=scenario.workload_name,
                device=device,
                sample_count=scenario.sample_count,
                batch_size=scenario.batch_size,
                resolution=scenario.resolution,
                scale=scenario.scale,
            )
            config = FunctionalEvalConfig(
                request=request,
                seed=seed,
                point_chunk_sizes=scenario.point_chunk_sizes,
                max_memory_fraction=scenario.max_memory_fraction,
                run_label=run_label,
            )
            # Warm-up runs: executed but not recorded (JIT / caching settle-in).
            for warmup_index in range(scenario.warmup_repeats):
                run_experiment(config, repeat=warmup_index)

            # R measured runs: recorded and aggregated here at the runner level.
            raw_run_summaries = [
                run_experiment(config, repeat=r)
                for r in range(scenario.repeats)
            ]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            scenario_results.append(
                {
                    "scenario": asdict(scenario),
                    "status": "error",
                    "error": error,
                    "workload": _workload_summary(scenario, definition),
                    "platform": None,
                    "config": None,
                    "candidate_summary": [],
                    "section_timings": {},
                    "paired_candidate_metrics": [],
                    "best_valid_candidate": None,
                    "best_valid_vmap_chunk_size": None,
                    "memory": _empty_memory_summary(),
                    "validation": {
                        "status": "error",
                        "all_validations_passed": None,
                    },
                    "record": {
                        "scenario": scenario.name,
                        "workload": scenario.workload_name,
                        "status": "error",
                        "reason": error,
                        "candidates": [],
                    },
                }
            )
            continue

        # Merge all measured runs into a single combined summary for downstream
        # helpers that expect a "runs" list keyed by repeat index.
        merged = _merge_run_summaries(raw_run_summaries)

        validation = _validation_status(merged)
        best = _best_valid_candidate_info(merged)
        candidate_statuses = {
            str(row.get("candidate")): row.get("status_counts", {})
            for row in merged.get("candidate_summary", [])
        }
        ok_count = sum(
            int(statuses.get("ok", 0))
            for statuses in candidate_statuses.values()
        )
        status = "completed" if ok_count > 0 else "error"

        paired_metrics = _paired_candidate_metrics(merged)

        # Every candidate is always reported — validity flags are informational
        # only and never suppress a result.
        section_timings = _section_timing_summary(merged)
        record: dict[str, Any] = {
            "scenario": scenario.name,
            "workload": scenario.workload_name,
            "status": status,
            "validation_status": validation["status"],
            "all_validations_passed": validation["all_validations_passed"],
            "candidate_status_counts": candidate_statuses,
            "section_timings": section_timings,
            "candidates": [
                {
                    "candidate": m["candidate"],
                    "paired_speedup_mean": m["paired_speedup_mean"],
                    "paired_speedup_stdev": m["paired_speedup_stdev"],
                    "paired_speedup_ci_95_lo": m["paired_speedup_ci_95_lo"],
                    "paired_speedup_ci_95_hi": m["paired_speedup_ci_95_hi"],
                    "paired_speedup_min": m["paired_speedup_min"],
                    "paired_speedup_max": m["paired_speedup_max"],
                    "all_repeats_beat_baseline": m["all_repeats_beat_baseline"],
                    "all_validations_passed": m["all_validations_passed"],
                    "all_within_budget": m["all_within_budget"],
                    "surface_budget": m["surface_budget"],
                    "claim_status": m["claim_status"],
                }
                for m in paired_metrics
            ],
            "best_valid_candidate": best,
        }
        if status == "error":
            record["reason"] = "no candidate completed with measured timings"

        scenario_results.append(
            {
                "scenario": asdict(scenario),
                "status": status,
                "workload": _workload_summary(scenario, definition),
                "platform": merged.get("platform"),
                "config": merged.get("config"),
                "candidate_summary": merged.get("candidate_summary", []),
                "section_timings": section_timings,
                "paired_candidate_metrics": paired_metrics,
                "best_valid_candidate": best,
                "best_valid_vmap_chunk_size": _best_valid_vmap_chunk_size_info(best),
                "memory": _memory_summary(merged),
                "validation": validation,
                "record": record,
                **({"error": record["reason"]} if status == "error" else {}),
            }
        )

    failed = [
        item["record"]
        for item in scenario_results
        if item.get("record", {}).get("status") == "error"
    ]
    record = {
        "status": "completed" if not failed else "completed_with_errors",
        "scenario_count": len(scenario_results),
        "completed_scenario_count": len(scenario_results) - len(failed),
        "failed_scenario_count": len(failed),
        "failed": failed,
    }
    suite = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "seed": seed,
        "run_label": run_label,
        "scenario_count": len(scenario_results),
        "scenarios": scenario_results,
        "record": record,
    }
    _print_suite_table(suite)
    return suite


def run_api_probe(
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
) -> dict[str, Any]:
    del output_dir
    del shared_state
    result = run_functional_api_probe(config.device, seed=config.seed)
    status = "completed" if not result.get("errors") else "completed_with_errors"
    record = {
        "status": status,
        "error_count": len(result.get("errors", []) or []),
        "skipped_count": len(result.get("skipped_checks", []) or []),
    }
    return {
        "status": status,
        "result": result,
        "record": record,
        "child_stem": "functional-eval-api-probe",
    }


def run_platform_benchmark(
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
) -> dict[str, Any]:
    del output_dir
    del shared_state
    result = run_platform_suite(
        scenarios=scenarios(config),
        device=config.device,
        seed=config.seed,
        run_label=config.run_label,
    )
    status = result["record"]["status"]
    return {
        "status": status,
        "result": result,
        "record": result["record"],
        "child_stem": "functional-eval-platform-suite",
    }


def scenarios(config: SimpleNamespace) -> tuple[PlatformSuiteScenario, ...]:
    items: list[PlatformSuiteScenario] = []
    workload_names = list(
        getattr(config, "functional_eval_workloads", None)
        or getattr(config, "mltask_workloads", None)
        or DEFAULT_FUNCTIONAL_EVAL_WORKLOADS
    )
    if config.include_vmap_reproduction:
        for workload_name in workload_names:
            items.append(
                PlatformSuiteScenario(
                    name=f"{workload_name}__vmap_reproduction",
                    sample_count=config.sample_count,
                    workload_name=workload_name,
                    repeats=config.functional_eval_repeats,
                    batch_size=config.functional_eval_batch_size,
                    resolution=config.grid_resolution,
                    scale=config.grid_scale,
                    point_chunk_sizes=(1, 2, 4, 8, 16, 32, 64),
                    max_memory_fraction=config.max_memory_fraction,
                )
            )

    sample_counts = [
        sample_count
        for sample_count in config.functional_eval_sample_counts
        if not (
            config.include_vmap_reproduction and sample_count == config.sample_count
        )
    ]
    items.extend(
        build_platform_scenarios(
            workload_names=workload_names,
            sample_counts=sample_counts,
            repeats=config.functional_eval_repeats,
            batch_size=config.functional_eval_batch_size,
            resolution=config.grid_resolution,
            scale=config.grid_scale,
            point_chunk_sizes=config.point_chunk_sizes,
            max_memory_fraction=config.max_memory_fraction,
            include_full_test_set=config.include_full_test_set,
        )
    )
    return tuple(items)


def default_scenarios() -> tuple[PlatformSuiteScenario, ...]:
    return DEFAULT_SCENARIOS


def build_platform_scenarios(
    *,
    workload_names: Iterable[str] = DEFAULT_FUNCTIONAL_EVAL_WORKLOADS,
    sample_counts: Iterable[int] = DEFAULT_FUNCTIONAL_EVAL_SAMPLE_COUNTS,
    repeats: int = 5,
    batch_size: int = 32,
    resolution: int = 8,
    scale: float = 1.0,
    point_chunk_sizes: Iterable[int] = (),
    max_memory_fraction: float | None = 0.85,
    include_full_test_set: bool = False,
) -> tuple[PlatformSuiteScenario, ...]:
    scenarios: list[PlatformSuiteScenario] = []
    chunks = tuple(point_chunk_sizes)
    for workload_name in workload_names:
        for sample_count in sample_counts:
            if sample_count == 0 and not include_full_test_set:
                continue
            scenario_name = (
                "full_test_set_scaling"
                if sample_count == 0
                else f"focused_{sample_count}_scaling"
            )
            scenarios.append(
                PlatformSuiteScenario(
                    name=f"{workload_name}__{scenario_name}",
                    sample_count=sample_count,
                    workload_name=workload_name,
                    repeats=repeats,
                    batch_size=batch_size,
                    resolution=resolution,
                    scale=scale,
                    point_chunk_sizes=chunks,
                    max_memory_fraction=max_memory_fraction,
                )
            )
    return tuple(scenarios)


def _workload_summary(
    scenario: PlatformSuiteScenario,
    definition: WorkloadDefinition | None,
) -> dict[str, Any]:
    if definition is None:
        return {
            "workload_name": scenario.workload_name,
            "registered": False,
            "sample_count": scenario.sample_count,
        }

    spec = definition.spec
    return {
        "workload_name": spec.name,
        "registered": True,
        "model_family": spec.model,
        "task": spec.task,
        "loss": spec.loss,
        "dataset": asdict(replace(spec.dataset, sample_count=scenario.sample_count)),
        "checkpoint_path": spec.checkpoint_path,
    }


def _merge_run_summaries(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-repeat single-run summaries into a single combined summary.

    Each entry in run_summaries is the output of one run_experiment() call.
    The repeat index is assigned sequentially (0 … R-1) so that
    _paired_candidate_metrics can match baseline and candidate runs by repeat.
    """
    all_runs: list[dict[str, Any]] = []
    platform: dict[str, Any] | None = None
    config: dict[str, Any] | None = None

    for repeat_idx, single in enumerate(run_summaries):
        if platform is None:
            platform = single.get("platform")
        if config is None:
            config = single.get("config")
        for run in single.get("runs", []):
            tagged = dict(run)
            tagged["repeat"] = repeat_idx
            all_runs.append(tagged)

    # Aggregate candidate_summary across all repeats.
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for run in all_runs:
        by_candidate.setdefault(run["candidate"], []).append(run)

    candidate_summary: list[dict[str, Any]] = []
    for candidate, runs in by_candidate.items():
        ok_times = [
            float(r["total_grid_s"])
            for r in runs
            if r.get("status") == "ok" and r.get("total_grid_s") is not None
        ]
        taxonomy = next(
            (r.get("metadata", {}).get("candidate_taxonomy") for r in runs if r.get("metadata")),
            None,
        )
        status_counts: dict[str, int] = {}
        for r in runs:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        candidate_summary.append(
            {
                "candidate": candidate,
                "taxonomy": taxonomy,
                "status_counts": status_counts,
                "mean_total_grid_s": _mean(ok_times),
                "stdev_total_grid_s": _stdev(ok_times),
            }
        )

    return {
        "platform": platform,
        "config": config,
        "runs": all_runs,
        "candidate_summary": candidate_summary,
    }


def _best_valid_candidate_info(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Return the candidate with the highest paired speedup that *also* clears
    the paired-CI rejection rule (CI_lo > 1.0) and the surface budget.

    A row failing surface validation, or whose 95% CI includes 1.0, is not a
    supported speedup and is excluded. The result is informational only and
    does not suppress any candidate's individual metrics.
    """
    candidates = _paired_candidate_metrics(summary)
    if not candidates:
        return None
    valid = [
        row
        for row in candidates
        if row.get("claim_status") == "speedup"
        and row.get("paired_speedup_mean") is not None
    ]
    if not valid:
        return None
    return max(valid, key=lambda row: float(row["paired_speedup_mean"]))


def _section_timing_summary(summary: dict[str, Any]) -> dict[str, Any]:
    by_candidate: dict[str, dict[str, list[float]]] = {}
    for run in summary.get("runs", []):
        if run.get("status") != "ok":
            continue
        timings = run.get("timings") or {}
        candidate = str(run.get("candidate"))
        buckets = by_candidate.setdefault(
            candidate,
            {
                "perturbation_s": [],
                "binding_s": [],
                "batch_eval_s": [],
                "total_grid_s": [],
            },
        )
        for key in buckets:
            value = timings.get(key)
            if value is not None:
                buckets[key].append(float(value))

    return {
        candidate: {
            key: _mean(values)
            for key, values in timings.items()
        }
        for candidate, timings in by_candidate.items()
    }


def _best_valid_vmap_chunk_size_info(best: dict[str, Any] | None) -> int | None:
    if best is None:
        return None
    prefix = "vmapped_chunk_"
    candidate = str(best["candidate"])
    if not candidate.startswith(prefix):
        return None
    try:
        return int(candidate.removeprefix(prefix))
    except ValueError:
        return None


def _memory_summary(summary: dict[str, Any]) -> dict[str, Any]:
    cpu_values = []
    cuda_values = []
    for run in summary.get("runs", []):
        if run.get("peak_cpu_memory_bytes") is not None:
            cpu_values.append(int(run["peak_cpu_memory_bytes"]))
        if run.get("peak_cuda_memory_bytes") is not None:
            cuda_values.append(int(run["peak_cuda_memory_bytes"]))
    return {
        "peak_cpu_memory_bytes": max(cpu_values) if cpu_values else None,
        "peak_cuda_memory_bytes": max(cuda_values) if cuda_values else None,
    }


def _empty_memory_summary() -> dict[str, Any]:
    return {
        "peak_cpu_memory_bytes": None,
        "peak_cuda_memory_bytes": None,
    }


def _validation_status(summary: dict[str, Any]) -> dict[str, Any]:
    validations = []
    for run in summary.get("runs", []):
        validation = run.get("validation")
        if validation is not None:
            validations.append(bool(validation.get("allclose")))
    return {
        "status": "checked" if validations else "not_applicable",
        "all_validations_passed": all(validations) if validations else None,
    }


_SURFACE_MAX_ABS_BUDGET = 1e-4  # float32 rounding budget per spec

# Two-tailed 95% Student-t critical values keyed by df = n - 1. Used so the CI
# rejection rule (CI_lo > 1 ⇒ speedup) is correct for any R, not only R = 5.
_T_CRIT_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
}


def _t_interval_95(values: list[float]) -> tuple[float, float] | None:
    """Return (lower, upper) 95% t-interval for the mean of values.

    Uses Student's t with df = n - 1. Requires at least 2 observations.
    Dispatches to the df-specific critical value so T4 (R=3, df=2, t=4.303)
    is not under-widened by an R=5 (df=4) constant.
    """
    n = len(values)
    if n < 2:
        return None
    t_crit = _T_CRIT_95.get(n - 1)
    if t_crit is None:
        return None
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half_width = t_crit * sd / n ** 0.5
    return mean - half_width, mean + half_width


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
        within_budget_flags = []
        for run in candidate_runs:
            baseline = baselines.get(int(run["repeat"]))
            if baseline is not None:
                paired_speedups.append(
                    float(baseline["total_grid_s"]) / float(run["total_grid_s"])
                )
            validation = run.get("validation")
            if validation is not None:
                validations.append(bool(validation.get("allclose")))
                within_budget_flags.append(
                    bool(validation.get("max_abs_within_budget", True))
                )

        ci = _t_interval_95(paired_speedups)
        # Informational flag: whether every run passes the max absolute deviation
        # budget (≤ 1e-4). Reported as-is; never used to suppress speedup output.
        all_within_budget = all(within_budget_flags) if within_budget_flags else None

        ci_lo = ci[0] if ci is not None else None
        ci_hi = ci[1] if ci is not None else None
        all_validations_passed = all(validations) if validations else None
        metrics.append(
            {
                "candidate": row["candidate"],
                "taxonomy": row.get("taxonomy"),
                "status_counts": row["status_counts"],
                "mean_total_grid_s": row.get("mean_total_grid_s"),
                "summary_mean_speedup_vs_baseline": row.get(
                    "mean_speedup_vs_baseline"
                ),
                "all_validations_passed": all_validations_passed,
                "all_within_budget": all_within_budget,
                "surface_budget": _SURFACE_MAX_ABS_BUDGET,
                "paired_speedups": paired_speedups,
                "paired_speedup_mean": _mean(paired_speedups),
                "paired_speedup_min": min(paired_speedups) if paired_speedups else None,
                "paired_speedup_max": max(paired_speedups) if paired_speedups else None,
                "paired_speedup_stdev": _stdev(paired_speedups),
                "paired_speedup_ci_95_lo": ci_lo,
                "paired_speedup_ci_95_hi": ci_hi,
                "all_repeats_beat_baseline": (
                    all(speedup >= 1.0 for speedup in paired_speedups)
                    if paired_speedups else None
                ),
                "claim_status": _claim_status(
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                    all_validations_passed=all_validations_passed,
                    all_within_budget=all_within_budget,
                ),
            }
        )
    return metrics


def _claim_status(
    *,
    ci_lo: float | None,
    ci_hi: float | None,
    all_validations_passed: bool | None,
    all_within_budget: bool | None,
) -> str:
    """Categorize a candidate row against the paired-CI rejection rule.

    Returns one of: "speedup", "inconclusive", "regression", "invalid_surface",
    "insufficient_data". Surface validity is a precondition: a row that fails
    surface validation is reported as "invalid_surface" regardless of timing.
    """
    if all_validations_passed is False or all_within_budget is False:
        return "invalid_surface"
    if ci_lo is None or ci_hi is None:
        return "insufficient_data"
    if ci_lo > 1.0:
        return "speedup"
    if ci_hi < 1.0:
        return "regression"
    return "inconclusive"


def _print_suite_table(suite: dict[str, Any]) -> None:
    print("functional_eval platform suite")
    print("scenario                 status     best_candidate            speedup")
    for item in suite["scenarios"]:
        best = item.get("best_valid_candidate")
        candidate = None if best is None else best["candidate"]
        speedup = None if best is None else best["paired_speedup_mean"]
        print(
            f"{item['scenario']['name']:<24} "
            f"{item['status']:<10.10} "
            f"{str(candidate):<25.25} "
            f"{_format_float(speedup):>8}"
        )


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


def build_config(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "device": "cuda",
        "seed": 1337,
        "repeats": 5,
        "batch_size": 32,
        "resolution": 8,
        "scale": 1.0,
        "sample_counts": [1024, 2048],
        "workload_name": DEFAULT_WORKLOAD_NAME,
        "workload_names": None,
        "point_chunk_sizes": [],
        "include_prd_confirmation": False,
        "include_full_test_set": False,
        "output_dir": str(Path("outputs") / "functional_eval" / "platform_suite"),
        "run_label": None,
    }
    defaults.update(overrides)
    if defaults["workload_names"] is None:
        defaults["workload_names"] = (
            [defaults["workload_name"]]
            if "workload_name" in overrides
            else list(DEFAULT_FUNCTIONAL_EVAL_WORKLOADS)
        )
    return SimpleNamespace(**defaults)


def main() -> None:
    args = build_config()
    scenarios: list[PlatformSuiteScenario] = []
    workload_names = list(args.workload_names)
    if args.include_prd_confirmation:
        for workload_name in workload_names:
            scenarios.append(
                replace(
                    PRD_VMAP_REPRODUCTION_SCENARIO,
                    name=f"{workload_name}__vmap_reproduction",
                    workload_name=workload_name,
                )
            )

    sample_counts = [
        sample_count
        for sample_count in args.sample_counts
        if not (args.include_prd_confirmation and sample_count == 1024)
    ]
    scenarios.extend(
        build_platform_scenarios(
            workload_names=workload_names,
            sample_counts=sample_counts,
            repeats=args.repeats,
            batch_size=args.batch_size,
            resolution=args.resolution,
            scale=args.scale,
            point_chunk_sizes=args.point_chunk_sizes,
            include_full_test_set=args.include_full_test_set,
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
