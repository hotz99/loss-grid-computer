from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.primitives import calibration_session
from experiments.common import unavailable_payload, workload_unavailable_reason
from src.backends import run_backend
from src.schemas import GridSpec, MLTaskSpec, SchedulerRequest, VanillaMode
from src.workloads import WORKLOADS


def _run_vanilla_reference_session(
    variant_tasks: list[MLTaskSpec],
    grid: GridSpec,
    gpu_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    session_start = time.perf_counter()
    per_variant_s: list[float] = []
    for index, task in enumerate(variant_tasks):
        variant_start = time.perf_counter()
        run_backend(
            SchedulerRequest(task, grid, VanillaMode(gpu_batch_size)),
            seed=seed + index,
            gpu_slowdown_factor=1.0,
        )
        per_variant_s.append(round(time.perf_counter() - variant_start, 3))
    session_total_s = round(time.perf_counter() - session_start, 3)
    return {
        "source": "empirical",
        "variants": len(variant_tasks),
        "session_grid_resolution": grid.resolution,
        "session_total_s": session_total_s,
        "per_variant_s": per_variant_s,
        "mean_per_variant_s": (
            round(sum(per_variant_s) / len(per_variant_s), 3) if per_variant_s else None
        ),
    }


def _compute_break_even(
    baseline_s: float | None,
    calibration_s: float | None,
    vanilla_per_variant_s: float | None,
    hybrid_per_variant_grid_s: float | None,
) -> tuple[int | None, str | None]:
    if None in (baseline_s, calibration_s, vanilla_per_variant_s, hybrid_per_variant_grid_s):
        return None, "missing_input_metric"
    per_variant_gain = vanilla_per_variant_s - hybrid_per_variant_grid_s
    if per_variant_gain <= 0:
        return None, "hybrid_per_variant_not_faster"
    return math.ceil((baseline_s + calibration_s) / per_variant_gain), None


def collect(
    config: SimpleNamespace,
    workload_name: str,
    variant_paths: list[str],
) -> dict[str, Any]:
    unavailable_reason = workload_unavailable_reason(workload_name)
    if unavailable_reason is not None:
        payload = unavailable_payload(workload_name, unavailable_reason, config.sample_count)
        payload["record"] = {
            "status": "skipped",
            "reason": payload.get("reason"),
        }
        return payload

    base = WORKLOADS[workload_name].spec
    task = replace(
        base,
        dataset=replace(base.dataset, sample_count=config.sample_count),
        checkpoint_path=variant_paths[0],
    )
    missing = [path for path in variant_paths if not Path(path).exists()]
    if missing:
        payload = unavailable_payload(
            workload_name,
            f"variant checkpoint assets are missing: {missing}",
            config.sample_count,
        )
        payload["record"] = {
            "status": "skipped",
            "reason": payload.get("reason"),
        }
        return payload

    # Experiment C uses unslowed runtime. Artificial GPU slowdown belongs to
    # Experiment B's parity probe only.
    slowdown = 1.0

    # SoTA-aligned headline metric requires empirically measuring the anti-pattern
    # cost (hybrid-without-cache). The derived projection is insufficient.
    if not config.measure_without_cache:
        raise ValueError(
            "calibration_cache: measure_without_cache must be True so the "
            "hybrid-without-cache anti-pattern penalty can be measured."
        )

    session_grid_resolution = getattr(
        config,
        "experiment_c_session_grid_resolution",
        None,
    ) or config.grid_resolution
    grid = GridSpec(session_grid_resolution, config.grid_scale)
    variant_tasks = [
        replace(task, checkpoint_path=checkpoint_path)
        for checkpoint_path in variant_paths
    ]

    result = calibration_session.main(
        slowdown,
        config.calibration_retry,
        variant_paths,
        base_workload=task,
        grid=grid,
        gpu_batch_size=config.gpu_batch_size,
        seed=config.seed,
        measure_without_cache=config.measure_without_cache,
    )
    result["runtime_condition"] = {
        "source": "unslowed",
        "slowdown": slowdown,
    }
    result["comparison_design"] = {
        "vanilla_reference": "gpu_only_full_session_no_calibration",
        "hybrid_with_cache": "calibrate_once_then_reuse_selected_policy",
        "hybrid_without_cache": "recalibrate_each_variant_then_execute_selected_policy",
    }

    vanilla_reference = _run_vanilla_reference_session(
        variant_tasks,
        grid,
        config.gpu_batch_size,
        config.seed,
    )
    result["vanilla_reference_session"] = vanilla_reference

    runtime = result.get("runtime_condition") or {}
    setup = result.get("setup") or {}
    with_cache = result.get("with_cache_session") or {}
    without_cache = result.get("without_cache_session") or {}
    variant_details = without_cache.get("variant_details") or []
    variant_count = (
        len(variant_details) if variant_details else without_cache.get("variants")
    ) or with_cache.get("variants")
    mode_tags = [
        (v.get("execution_mode") or {}).get("_tag") for v in variant_details
    ]

    vanilla_session_total_s = vanilla_reference.get("session_total_s")
    with_cache_session_total_s = with_cache.get("session_total_s")
    without_cache_session_total_s = without_cache.get("session_total_s")

    session_speedup_vs_vanilla = (
        round(vanilla_session_total_s / with_cache_session_total_s, 4)
        if vanilla_session_total_s and with_cache_session_total_s
        else None
    )
    hybrid_without_cache_penalty_vs_vanilla = (
        round(without_cache_session_total_s / vanilla_session_total_s, 4)
        if without_cache_session_total_s and vanilla_session_total_s
        else None
    )

    vanilla_per_variant_s = vanilla_reference.get("mean_per_variant_s")
    hybrid_per_variant_grid_s = (
        round(with_cache.get("execution_s") / variant_count, 4)
        if with_cache.get("execution_s") is not None and variant_count
        else None
    )
    amortized_calibration_per_variant_s = (
        round(with_cache.get("calibration_s") / variant_count, 4)
        if with_cache.get("calibration_s") is not None and variant_count
        else None
    )

    break_even_n, break_even_reason = _compute_break_even(
        with_cache.get("baseline_s"),
        with_cache.get("calibration_s"),
        vanilla_per_variant_s,
        hybrid_per_variant_grid_s,
    )

    result["record"] = {
        "status": "completed",
        "slowdown": runtime.get("slowdown"),
        "slowdown_source": runtime.get("source"),
        "selected_policy": (with_cache.get("execution_mode") or {}).get("_tag"),
        "calibration_s": with_cache.get("calibration_s"),
        "calibration_grid_resolution": setup.get("calibration_grid_resolution"),
        "session_grid_resolution": setup.get("execution_grid_resolution"),
        "variant_count": variant_count,
        "mode_selections_by_variant": mode_tags if mode_tags else None,
        # SoTA-aligned headline + supporting metrics
        "session_total_s_vanilla": vanilla_session_total_s,
        "session_total_s_with_cache": with_cache_session_total_s,
        "session_total_s_without_cache": without_cache_session_total_s,
        "session_speedup_vs_vanilla": session_speedup_vs_vanilla,
        "hybrid_without_cache_penalty_vs_vanilla": hybrid_without_cache_penalty_vs_vanilla,
        "break_even_n": break_even_n,
        "break_even_unavailable_reason": break_even_reason,
        "amortized_calibration_per_variant_s": amortized_calibration_per_variant_s,
        "vanilla_per_variant_s": vanilla_per_variant_s,
        "hybrid_per_variant_grid_s": hybrid_per_variant_grid_s,
    }
    return result


def run(
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
    workload_name: str,
    variant_paths: list[str],
) -> dict[str, Any]:
    del output_dir
    del shared_state
    result = collect(config, workload_name, variant_paths)
    status = result.get("status", "completed") if isinstance(result, dict) else "completed"
    payload = {
        "status": status,
        "result": result,
        "record": result["record"],
        "child_stem": "experiment-c-calibration-cache",
    }
    if status == "skipped":
        payload["reason"] = (
            result.get("reason")
            if isinstance(result, dict)
            else "experiment returned skipped without details"
        ) or "experiment returned skipped without details"
    return payload


VARIANT_PATHS: dict[str, list[str]] = {
    "cifar10_resnet20_classification": [
        f"assets/cifar10-resnet20-{seed}.pkl" for seed in [0, 123, 2023, 123456]
    ],
    "cifar10_row_gru_classification": [
        f"assets/cifar10-row-gru-{seed}.pkl" for seed in [0, 42, 99, 1337]
    ],
    "california_mlp_regression": [
        f"assets/california-mlp-{seed}.pkl" for seed in [0, 42, 99, 1337]
    ],
    "mnist_mlp_classification": [
        f"assets/mnist-mlp-{seed}.pkl" for seed in [0, 42, 99, 1337]
    ],
}


def run_all(
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
) -> dict[str, Any]:
    del output_dir
    workload_results: dict[str, Any] = {}
    workload_records: dict[str, Any] = {}
    save_intermediate = shared_state.get("_save_intermediate_payload")
    for workload_name, rel_paths in VARIANT_PATHS.items():
        abs_paths = [
            str(Path(__file__).resolve().parents[1] / p) for p in rel_paths
        ]
        result = collect(config, workload_name, abs_paths)
        workload_results[workload_name] = result
        workload_records[workload_name] = result.get("record") or {}

        if callable(save_intermediate):
            completed = [
                name
                for name, record in workload_records.items()
                if record.get("status") == "completed"
            ]
            skipped = [
                name
                for name, record in workload_records.items()
                if record.get("status") == "skipped"
            ]
            save_intermediate(
                "experiment-c-calibration-cache-partial",
                {
                    "status": "running",
                    "result": {"workloads": workload_results},
                    "record": {
                        "status": "running",
                        "workloads": workload_records,
                        "completed_workloads": completed,
                        "skipped_workloads": skipped,
                    },
                    "child_stem": "experiment-c-calibration-cache-partial",
                },
            )

    completed = [
        name for name, r in workload_records.items() if r.get("status") == "completed"
    ]
    skipped = [
        name for name, r in workload_records.items() if r.get("status") == "skipped"
    ]
    status = "completed" if completed else "skipped"
    return {
        "status": status,
        "result": {"workloads": workload_results},
        "record": {
            "status": status,
            "workloads": workload_records,
            "completed_workloads": completed,
            "skipped_workloads": skipped,
        },
        "child_stem": "experiment-c-calibration-cache",
    }
