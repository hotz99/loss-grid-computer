from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.primitives import calibration_session
from experiments.common import unavailable_payload, workload_unavailable_reason
from src.schemas import GridSpec
from src.workloads import WORKLOADS


def collect(config: SimpleNamespace, variant_paths: list[str]) -> dict[str, Any]:
    workload_name = "cifar10_resnet20_classification"
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
    result = calibration_session.main(
        slowdown,
        config.calibration_retry,
        variant_paths,
        base_workload=task,
        grid=GridSpec(config.grid_resolution, config.grid_scale),
        gpu_batch_size=config.gpu_batch_size,
        seed=config.seed,
        measure_without_cache=config.measure_without_cache,
    )
    result["runtime_condition"] = {
        "source": "unslowed",
        "slowdown": slowdown,
    }
    runtime = result.get("runtime_condition") or {}
    setup = result.get("setup") or {}
    with_cache = result.get("with_cache_session") or {}
    without_cache = result.get("without_cache_session") or {}
    variant_details = without_cache.get("variant_details") or []
    variant_count = len(variant_details) if variant_details else without_cache.get("variants")
    mode_tags = [
        (v.get("execution_mode") or {}).get("_tag")
        for v in variant_details
    ]
    _mode_varies = (
        bool(len(set(t for t in mode_tags if t is not None)) > 1)
        if mode_tags else None
    )
    mode_selection_consistent = (
        bool(not _mode_varies)
        if _mode_varies is not None
        else None
    )
    result["record"] = {
        "status": "completed",
        "slowdown": runtime.get("slowdown"),
        "slowdown_source": runtime.get("source"),
        "selected_policy": (with_cache.get("execution_mode") or {}).get("_tag"),
        "calibration_s": with_cache.get("calibration_s"),
        "calibration_grid_resolution": setup.get("calibration_grid_resolution"),
        "session_total_s_with_cache": with_cache.get("session_total_s"),
        "session_total_s_without_cache": without_cache.get("session_total_s"),
        "best_hybrid_minus_gpu_only_s": with_cache.get("best_hybrid_minus_gpu_only_s"),
        "variant_count": variant_count,
        "mode_selections_by_variant": mode_tags if mode_tags else None,
        "mode_selection_consistent": mode_selection_consistent,
        "time_saved_s": (
            round(without_cache.get("session_total_s") - with_cache.get("session_total_s"), 4)
            if without_cache.get("session_total_s") is not None and with_cache.get("session_total_s") is not None
            else None
        ),
    }
    return result


def run(
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
    variant_paths: list[str],
) -> dict[str, Any]:
    del output_dir
    del shared_state
    result = collect(config, variant_paths)
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
