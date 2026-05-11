from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.common import (
    configured_workloads,
    get_shared_artifact,
    shared_artifact_key,
    unavailable_payload,
    workload_metadata,
    workload_unavailable_reason,
)
from src.backends import run_backend
from src.calibration import (
    resolve_available_cpu_cores,
    resolve_cpu_batch_size_candidates,
    resolve_cpu_worker_candidates,
)
from src.compare import compare_surfaces
from src.schemas import GridSpec, HybridMode, MLTaskSpec, SchedulerRequest, VanillaMode
from src.workloads import WORKLOADS


def measurement_total_s(result: Any) -> float:
    timings = getattr(result, "timings", None)
    if timings is not None:
        return float(timings.total_grid_s)
    return float(result.record.measurement.total_s)


def measurement_throughput(result: Any) -> float:
    timings = getattr(result, "timings", None)
    records = getattr(result, "records", None)
    if timings is not None and records is not None:
        total_s = float(timings.total_grid_s)
        return 0.0 if total_s <= 0 else len(records) / total_s
    execution = result.runtime_log.get("vanilla_execution", {})
    throughput = execution.get("throughput_points_per_s")
    if throughput is not None:
        return float(throughput)
    return float(result.record.measurement.get_points_per_s)


def _mode_kind(mode: Any) -> str:
    if isinstance(mode, VanillaMode):
        return "gpu_only"
    if isinstance(mode, HybridMode):
        return "gpu_cpu_hybrid"
    return type(mode).__name__


def _worker_split(result: Any) -> dict[str, Any] | None:
    execution = result.runtime_log.get("hybrid_execution")
    if not execution:
        return None
    gpu_points = int(execution.get("gpu_points_processed", 0))
    cpu_points = int(execution.get("cpu_points_processed", 0))
    total_points = gpu_points + cpu_points
    return {
        "gpu_points_processed": gpu_points,
        "cpu_points_processed": cpu_points,
        "gpu_point_fraction": None if total_points <= 0 else gpu_points / total_points,
        "cpu_point_fraction": None if total_points <= 0 else cpu_points / total_points,
        "gpu_throughput_points_per_s": execution.get("gpu_throughput_points_per_s"),
        "cpu_throughput_points_per_s": execution.get("cpu_throughput_points_per_s"),
        "total_throughput_points_per_s": execution.get("throughput_points_per_s"),
    }


def _cpu_worker_candidates(config: SimpleNamespace) -> tuple[int, ...]:
    candidates = resolve_cpu_worker_candidates()
    max_candidate = getattr(config, "max_cpu_worker_candidate", None)
    if max_candidate is None:
        return candidates
    capped = tuple(value for value in candidates if value <= int(max_candidate))
    return capped or (min(candidates),)


def _shared_vanilla_full_grid(
    *,
    task: MLTaskSpec,
    grid: GridSpec,
    config: SimpleNamespace,
    shared_state: dict[str, Any],
) -> Any | None:
    key = shared_artifact_key(
        node_type="vanilla_full_grid",
        workload_name=task.name,
        checkpoint_path=task.checkpoint_path,
        sample_count=task.dataset.sample_count,
        grid_resolution=grid.resolution,
        grid_scale=grid.scale,
        device=config.device,
        seed=config.seed,
        gpu_batch_size=config.gpu_batch_size,
        slowdown_factor=1.0,
    )
    return get_shared_artifact(shared_state, key)


def _calibrate_with_result(
    *,
    task: MLTaskSpec,
    grid: GridSpec,
    config: SimpleNamespace,
    baseline_total_s: float,
    slowdown: float,
) -> tuple[VanillaMode | HybridMode, Any | None, list[dict[str, Any]]]:
    calibration_runs: list[dict[str, Any]] = []
    min_total_across_worker_counts = baseline_total_s
    consecutive_non_improvements = 0
    best_total_s: float | None = None
    best_mode: HybridMode | None = None
    best_result: Any | None = None

    cpu_worker_values = _cpu_worker_candidates(config)
    cpu_batch_sizes = resolve_cpu_batch_size_candidates(
        task.dataset.sample_count,
        config.gpu_batch_size,
    )

    for cpu_workers in cpu_worker_values:
        min_total_for_worker_count: float | None = None

        for cpu_batch_size in cpu_batch_sizes:
            candidate_mode = HybridMode(
                config.gpu_batch_size,
                cpu_batch_size,
                cpu_workers,
            )
            result = run_backend(
                SchedulerRequest(task, grid, candidate_mode, config.device),
                config.seed,
                slowdown,
            )
            total_s = measurement_total_s(result)
            calibration_runs.append(
                {
                    "cpu_workers": cpu_workers,
                    "cpu_batch_size": cpu_batch_size,
                    "total_s": total_s,
                }
            )
            if best_total_s is None or total_s < best_total_s:
                best_total_s = total_s
                best_mode = candidate_mode
                best_result = result

            if min_total_for_worker_count is None or total_s < min_total_for_worker_count:
                min_total_for_worker_count = total_s

            if min_total_for_worker_count < min_total_across_worker_counts:
                min_total_across_worker_counts = min_total_for_worker_count
                consecutive_non_improvements = 0
                continue

            consecutive_non_improvements += 1
            if consecutive_non_improvements >= config.calibration_retry:
                break

    if best_total_s is None or best_mode is None or best_total_s >= baseline_total_s:
        return VanillaMode(config.gpu_batch_size), None, calibration_runs
    return best_mode, best_result, calibration_runs


def _unslowed_ratio(
    *,
    task: MLTaskSpec,
    grid: GridSpec,
    config: SimpleNamespace,
    cpu_batch_size: int,
    cpu_workers: int,
    vanilla_result: Any,
) -> dict[str, Any]:
    cpu_result = run_backend(
        SchedulerRequest(
            task,
            grid,
            HybridMode(config.gpu_batch_size, cpu_batch_size, cpu_workers),
            "cpu",
        ),
        config.seed,
    )
    gpu_throughput = measurement_throughput(vanilla_result)
    cpu_throughput = measurement_throughput(cpu_result)
    return {
        "gpu_runtime_s": measurement_total_s(vanilla_result),
        "cpu_runtime_s": measurement_total_s(cpu_result),
        "gpu_throughput_points_per_s": gpu_throughput,
        "cpu_throughput_points_per_s": cpu_throughput,
        "cpu_gpu_inference_ratio": None
        if gpu_throughput <= 0
        else cpu_throughput / gpu_throughput,
        "cpu_workers": cpu_workers,
        "cpu_batch_size": cpu_batch_size,
    }


def _evaluate_ratio_regime(
    *,
    task: MLTaskSpec,
    grid: GridSpec,
    config: SimpleNamespace,
    native_vanilla: Any,
    unslowed: dict[str, Any],
    target_ratio: float,
    slowdown: float,
) -> dict[str, Any]:
    fixed_vanilla = _slowdown_adjusted_vanilla_result(native_vanilla, slowdown)
    selected_mode, calibrated_result, calibration_runs = _calibrate_with_result(
        task=task,
        grid=grid,
        config=config,
        baseline_total_s=measurement_total_s(fixed_vanilla),
        slowdown=slowdown,
    )
    selected_result = (
        fixed_vanilla
        if isinstance(selected_mode, VanillaMode)
        else calibrated_result
    )
    if selected_result is None:
        raise RuntimeError("calibration selected hybrid mode without a result")
    surface = compare_surfaces(
        fixed_vanilla.records,
        selected_result.records,
        config.atol,
        config.rtol,
        measurement_total_s(fixed_vanilla),
        measurement_total_s(selected_result),
    )
    slowed_gpu_throughput = measurement_throughput(fixed_vanilla)
    cpu_throughput = unslowed["cpu_throughput_points_per_s"]
    achieved_ratio = (
        None
        if slowed_gpu_throughput <= 0
        else cpu_throughput / slowed_gpu_throughput
    )
    speedup = surface["speedup_rhs_vs_lhs_baseline"]
    surface_valid = bool(surface["allclose"])
    return {
        "target_cpu_gpu_inference_ratio": target_ratio,
        "unslowed_cpu_gpu_inference_ratio": unslowed["cpu_gpu_inference_ratio"],
        "achieved_cpu_gpu_inference_ratio": achieved_ratio,
        "ratio_error": None if achieved_ratio is None else achieved_ratio - target_ratio,
        "ratio_error_abs": None
        if achieved_ratio is None
        else abs(achieved_ratio - target_ratio),
        "slowdown": slowdown,
        "status": "completed",
        "selected_policy": _mode_kind(selected_mode),
        "selected_mode": asdict(selected_mode),
        "vanilla_runtime_s": measurement_total_s(fixed_vanilla),
        "slowed_gpu_throughput_points_per_s": slowed_gpu_throughput,
        "cpu_throughput_points_per_s": cpu_throughput,
        "selected_runtime_s": measurement_total_s(selected_result),
        "calibration_candidate_count": len(calibration_runs),
        "speedup_vs_vanilla": speedup,
        "hybrid_wins": (
            isinstance(selected_mode, HybridMode)
            and surface_valid
            and speedup is not None
            and speedup > 1.0
        ),
        "surface_valid": surface_valid,
        "surface_validation": {
            "allclose": surface["allclose"],
            "rmse": surface["rmse"],
            "mismatch_count": surface["mismatch_count"],
            "atol": surface["atol"],
            "rtol": surface["rtol"],
            "max_abs_error": surface.get("max_abs_error"),
        },
        "worker_throughput_split": _worker_split(selected_result),
    }


def _slowdown_adjusted_vanilla_result(result: Any, slowdown: float) -> Any:
    if slowdown == 1.0:
        return result

    native_total_s = measurement_total_s(result)
    slowed_total_s = native_total_s * slowdown
    point_count = len(getattr(result, "records", []))
    slowed_throughput = 0.0 if slowed_total_s <= 0 else point_count / slowed_total_s
    runtime_log = dict(result.runtime_log)
    vanilla_execution = dict(runtime_log.get("vanilla_execution") or {})
    vanilla_execution.update(
        {
            "grid_compute_only_s": slowed_total_s,
            "throughput_points_per_s": slowed_throughput,
            "slowdown_source": "derived_from_profiled_vanilla",
            "native_grid_compute_only_s": native_total_s,
            "gpu_slowdown_factor": slowdown,
        }
    )
    runtime_log["total_s"] = slowed_total_s
    runtime_log["vanilla_execution"] = vanilla_execution
    measurement = replace(
        result.record.measurement,
        total_s=float(slowed_total_s),
        num_points=point_count,
    )
    record = replace(result.record, measurement=measurement)
    return replace(result, record=record, runtime_log=runtime_log)


def _parity_probe(
    *,
    task: MLTaskSpec,
    grid: GridSpec,
    config: SimpleNamespace,
    native_vanilla: Any,
    unslowed: dict[str, Any],
    slowdown: float,
) -> dict[str, Any]:
    regime = _evaluate_ratio_regime(
        task=task,
        grid=grid,
        config=config,
        native_vanilla=native_vanilla,
        unslowed=unslowed,
        target_ratio=1.0,
        slowdown=slowdown,
    )
    return {
        "target_ratio": 1.0,
        "slowdown_used": slowdown,
        "achieved_ratio": regime["achieved_cpu_gpu_inference_ratio"],
        "selected_policy": regime["selected_policy"],
        "selected_mode": regime["selected_mode"],
        "vanilla_runtime_s": regime["vanilla_runtime_s"],
        "selected_runtime_s": regime["selected_runtime_s"],
        "speedup_vs_vanilla": regime["speedup_vs_vanilla"],
        "hybrid_wins": regime["hybrid_wins"],
        "surface_valid": regime["surface_valid"],
        "surface_validation": regime["surface_validation"],
        "worker_throughput_split": regime["worker_throughput_split"],
    }


def _workload_result(
    *,
    task: MLTaskSpec,
    grid: GridSpec,
    config: SimpleNamespace,
    cpu_workers: int,
    shared_state: dict[str, Any],
) -> dict[str, Any]:
    cpu_batch_size = min(4, config.gpu_batch_size)
    vanilla_result = _shared_vanilla_full_grid(
        task=task,
        grid=grid,
        config=config,
        shared_state=shared_state,
    )
    vanilla_source = "precomputed"
    if vanilla_result is None:
        vanilla_result = run_backend(
            SchedulerRequest(
                task,
                grid,
                VanillaMode(config.gpu_batch_size),
                config.device,
            ),
            config.seed,
        )
        vanilla_source = "computed"
    unslowed = _unslowed_ratio(
        task=task,
        grid=grid,
        config=config,
        cpu_batch_size=cpu_batch_size,
        cpu_workers=cpu_workers,
        vanilla_result=vanilla_result,
    )
    unslowed["gpu_baseline_source"] = vanilla_source
    unslowed_ratio = unslowed["cpu_gpu_inference_ratio"]
    slowdown = 1.0 if unslowed_ratio is None or unslowed_ratio >= 1.0 else 1.0 / unslowed_ratio
    return {
        "unslowed_ratio": unslowed,
        "parity_probe": _parity_probe(
            task=task,
            grid=grid,
            config=config,
            native_vanilla=vanilla_result,
            unslowed=unslowed,
            slowdown=slowdown,
        ),
    }


def collect(config: SimpleNamespace, shared_state: dict[str, Any] | None = None) -> dict[str, Any]:
    if shared_state is None:
        shared_state = {}
    workloads: dict[str, Any] = {}
    cpu_workers = resolve_available_cpu_cores()
    grid = GridSpec(config.grid_resolution, config.grid_scale)

    for workload_name in configured_workloads(config):
        unavailable_reason = workload_unavailable_reason(workload_name)
        if unavailable_reason is not None:
            workloads[workload_name] = unavailable_payload(
                workload_name,
                unavailable_reason,
                config.sample_count,
            )
            continue

        definition = WORKLOADS[workload_name]
        task = replace(
            definition.spec,
            dataset=replace(definition.spec.dataset, sample_count=config.sample_count),
        )
        try:
            workloads[workload_name] = {
                "status": "completed",
                "workload": workload_metadata(workload_name, config.sample_count),
                "result": _workload_result(
                    task=task,
                    grid=grid,
                    config=config,
                    cpu_workers=cpu_workers,
                    shared_state=shared_state,
                ),
            }
        except Exception as exc:
            workloads[workload_name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "workload": workload_metadata(workload_name, config.sample_count),
            }

    compact: dict[str, Any] = {}
    for name, item in workloads.items():
        if item.get("status") != "completed":
            compact[name] = {
                "status": item.get("status"),
                "reason": item.get("reason"),
                "error": item.get("error"),
            }
            continue

        result = item.get("result") or {}
        unslowed = result.get("unslowed_ratio") or {}
        parity = result.get("parity_probe") or {}
        compact[name] = {
            "status": item.get("status"),
            "unslowed_inference_ratio": unslowed.get("cpu_gpu_inference_ratio"),
            "gpu_throughput_points_per_s": unslowed.get("gpu_throughput_points_per_s"),
            "cpu_throughput_points_per_s": unslowed.get("cpu_throughput_points_per_s"),
            "gpu_baseline_source": unslowed.get("gpu_baseline_source"),
            "parity_probe": {
                "slowdown_used": parity.get("slowdown_used"),
                "achieved_ratio": parity.get("achieved_ratio"),
                "selected_policy": parity.get("selected_policy"),
                "speedup": parity.get("speedup_vs_vanilla"),
                "vanilla_runtime_s": parity.get("vanilla_runtime_s"),
                "selected_runtime_s": parity.get("selected_runtime_s"),
                "surface_valid": parity.get("surface_valid"),
                "hybrid_wins": parity.get("hybrid_wins"),
                "worker_throughput_split": parity.get("worker_throughput_split"),
                "max_abs_error": (parity.get("surface_validation") or {}).get("max_abs_error"),
            },
        }
    return {
        "schema_version": "experiment-b-hybrid-applicability-v2",
        "control": {
            "calibration_retry": config.calibration_retry,
            "cpu_workers": cpu_workers,
        },
        "workloads": workloads,
        "record": {
            "status": "completed",
            "workloads": compact,
        },
    }


def run(
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
) -> dict[str, Any]:
    del output_dir
    result = collect(config, shared_state)
    return {
        "status": "completed",
        "result": result,
        "record": result["record"],
        "child_stem": "experiment-b-hybrid-applicability",
    }
