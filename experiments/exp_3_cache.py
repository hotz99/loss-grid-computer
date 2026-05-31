from __future__ import annotations

import math
import re
from dataclasses import replace as _replace
from typing import Any

import torch

from experiments import calibration as calibration_mod
from experiments import device as device_mod
from experiments import sessions as sessions_mod
from experiments.candidates import GpuCandidate, run_standalone
from experiments.data import list_same_family_checkpoints
from experiments.schemas import (
    Experiment1Result,
    Experiment2Result,
    Experiment3Config,
    Experiment3Result,
    GridSpec,
)
from experiments.surface_gate import validate_surface
from experiments.workloads import WORKLOADS, task_for_workload, workload_metadata


_SCHEMA_VERSION = "experiment-3-cache-v1"
_N_CHECKPOINTS = 4
_CANDIDATE_K_RE = re.compile(r"_k(\d+)$")
_COMPILING_ROLES = {"compiled", "compiled_vmapped"}


def run(
    config: Experiment3Config,
    experiment_1: Experiment1Result,
    experiment_2: Experiment2Result,
) -> Experiment3Result:
    """Warm-pool calibration cache for the RQ1 winner at RQ2's operating point."""
    _progress("start")
    selection = _select_from_experiment_2(experiment_2)
    if selection is None:
        _progress("skip", reason="no_hybrid_affinity")
        return _skipped(
            config, experiment_1, "no_hybrid_affinity",
            {"hybrid_label": "refuted", "rq2_selection": None},
        )
    workload_name = selection["workload_name"]
    slowdown = selection["slowdown_factor"]
    operating_point = selection["operating_point"]
    _progress(
        "selected", workload=workload_name,
        threshold_status=selection["threshold_status"],
        slowdown=slowdown, operating_point=operating_point,
    )
    if workload_name not in WORKLOADS:
        _progress("skip", workload=workload_name, reason="unknown_workload")
        return _skipped(
            config, experiment_1, "unknown_workload", {}, workload_name=workload_name,
        )

    a_config_name = _resolve_rq3_config(experiment_1, workload_name)
    a_config = _parse_gpu_candidate(a_config_name)
    point_chunk_size_k = _point_chunk_size_for(a_config)
    _progress(
        "a_config", workload=workload_name,
        candidate=a_config_name, point_chunk_size_K=point_chunk_size_k,
    )

    device = device_mod.resolve(config.device)
    task = task_for_workload(workload_name, config.sample_count)
    try:
        checkpoints = list_same_family_checkpoints(task, _N_CHECKPOINTS)
    except FileNotFoundError as exc:
        _progress("skip", workload=workload_name, reason="missing_checkpoints")
        return _skipped(
            config, experiment_1, "missing_checkpoints", {"error": str(exc)},
            workload_name=workload_name,
        )
    _progress("checkpoints", workload=workload_name, count=len(checkpoints), device=device.type)

    workers = calibration_mod.cpu_worker_candidates(config.max_cpu_worker_candidate)
    probe_grid = _probe_grid_for(max(workers), point_chunk_size_k)

    task_at_ckpt0 = _replace(task, checkpoint_path=checkpoints[0])
    _progress(
        "calibration_baseline",
        workload=workload_name,
        grid_resolution=probe_grid.resolution,
    )
    calibration_baseline = run_standalone(
        a_config, task_at_ckpt0, probe_grid,
        batch_size=config.gpu_batch_size, device=device, seed=config.seed,
        gpu_slowdown_factor=slowdown,
    )
    batches = calibration_mod.cpu_batch_size_candidates(
        task.dataset.sample_count, config.gpu_batch_size,
    )
    _progress("calibration", workload=workload_name, cpu_worker_candidates=len(workers))
    cached_cell = calibration_mod.calibrate(
        task_at_ckpt0, probe_grid,
        gpu_batch_size=config.gpu_batch_size,
        baseline_total_s=calibration_baseline.total_grid_s,
        cpu_workers=workers, cpu_batch_sizes=batches,
        patience=config.calibration_retry,
        device=device, seed=config.seed,
        gpu_slowdown_factor=slowdown,
        gpu_candidate=a_config,
    )
    if cached_cell.max_hybrid_cpu_points <= 0:
        _progress("calibration_starvation", workload=workload_name)
        return _calibration_starvation(
            config,
            experiment_1,
            experiment_2,
            workload_name=workload_name,
            selection=selection,
            rq3_config=a_config_name,
            point_chunk_size_k=point_chunk_size_k,
            probe_grid=probe_grid,
            cached_cell=cached_cell,
        )

    compile_s = _measure_compile_cost(
        a_config, task_at_ckpt0, probe_grid,
        gpu_batch_size=config.gpu_batch_size, device=device,
        seed=config.seed, gpu_slowdown_factor=slowdown,
    )
    _progress("compile_cost", workload=workload_name, candidate=a_config_name, compile_s=compile_s)

    _progress("vanilla_session", workload=workload_name, checkpoints=len(checkpoints))
    vanilla = sessions_mod.vanilla_session(
        task, config.session_grid, checkpoints,
        batch_size=config.gpu_batch_size, device=device, seed=config.seed,
        gpu_slowdown_factor=slowdown,
    )
    _progress("gpu_only_session", workload=workload_name, checkpoints=len(checkpoints))
    gpu_only = sessions_mod.gpu_only_session(
        task, config.session_grid, checkpoints,
        gpu_candidate=a_config, batch_size=config.gpu_batch_size,
        device=device, seed=config.seed, gpu_slowdown_factor=slowdown,
    )
    hybrid = None
    pool_startup_s = 0.0
    if cached_cell.selected_policy == "gpu_cpu_hybrid":
        _progress("hybrid_pool_session", workload=workload_name, checkpoints=len(checkpoints))
        hybrid, pool_startup_s = sessions_mod.hybrid_pool_session(
            task, config.session_grid, checkpoints,
            cell=cached_cell, gpu_candidate=a_config, device=device, seed=config.seed,
            gpu_slowdown_factor=slowdown,
        )

    _progress("validate", workload=workload_name)
    surface_validation = {
        "vanilla": _reference_surface_validations(vanilla),
        "gpu_only": _validate_session_surfaces(
            gpu_only, vanilla, arm="gpu_only", surface_gate=config.surface_gate,
        ),
        "hybrid": (
            _validate_session_surfaces(
                hybrid, vanilla, arm="hybrid", surface_gate=config.surface_gate,
            )
            if hybrid is not None
            else None
        ),
    }
    gpu_only_surface_valid = _surface_valid(surface_validation["gpu_only"])
    hybrid_surface_valid = (
        _surface_valid(surface_validation["hybrid"])
        if surface_validation["hybrid"] is not None
        else True
    )

    t_vanilla = vanilla.mean_t_grid_s
    t_gpu_only = gpu_only.mean_t_grid_s
    t_hybrid = hybrid.mean_t_grid_s if hybrid is not None else None
    break_even_compile = _break_even(compile_s, t_vanilla - t_gpu_only)
    compile_label = _amortization_label(break_even_compile)
    if t_hybrid is not None:
        break_even_hybrid = _break_even(
            cached_cell.calibration_s + pool_startup_s,
            t_gpu_only - t_hybrid,
        )
        hybrid_label = _amortization_label(break_even_hybrid)
    else:
        break_even_hybrid = math.inf
        hybrid_label = "refuted"

    selected_policy = cached_cell.selected_policy
    if selected_policy == "gpu_cpu_hybrid" and t_hybrid is not None:
        deployed_session_total_s = (
            compile_s + cached_cell.calibration_s + pool_startup_s
            + (_N_CHECKPOINTS * t_hybrid)
        )
        deployed_surface_valid = gpu_only_surface_valid and hybrid_surface_valid
    else:
        deployed_session_total_s = (
            compile_s + cached_cell.calibration_s + (_N_CHECKPOINTS * t_gpu_only)
        )
        deployed_surface_valid = gpu_only_surface_valid
    session_speedup = (
        (_N_CHECKPOINTS * t_vanilla) / deployed_session_total_s
        if deployed_session_total_s > 0 and deployed_surface_valid
        else None
    )

    surface_pairs = _session_surface_pairs(gpu_only, vanilla, arm="gpu_only")
    if hybrid is not None:
        surface_pairs.extend(_session_surface_pairs(hybrid, vanilla, arm="hybrid"))

    record = {
        "status": "completed",
        "implementation_status": "completed",
        "device": device.type,
        "workload": workload_name,
        "session_regime": {
            "workload": workload_name,
            "platform": device.type,
            "regime": selection["threshold_status"],
            "slowdown_factor": slowdown,
            "operating_point": operating_point,
            "cpu_pool_size": cached_cell.cpu_workers or 0,
            "r_native": selection.get("r_native"),
            "source_exp_a_record": experiment_1.schema_version,
            "source_exp_b_record": experiment_2.schema_version,
            "rq3_config": a_config_name,
        },
        "operating_point": operating_point,
        "slowdown_factor": slowdown,
        "rq2_selection": selection,
        "rq3_config": a_config_name,
        "rq3_config_compiles": a_config.role in _COMPILING_ROLES,
        "a_config_compiles": a_config.role in _COMPILING_ROLES,
        "selected_policy": selected_policy,
        "selected_b_cell": cached_cell.__dict__,
        "calibration_s": cached_cell.calibration_s,
        "compile_cold_start_s": compile_s,
        "pool_startup_s": pool_startup_s,
        "n_checkpoints": _N_CHECKPOINTS,
        "session_grid_resolution": config.session_grid.resolution,
        "calibration_grid_resolution": probe_grid.resolution,
        "point_chunk_size_K": point_chunk_size_k,
        "T_vanilla": t_vanilla,
        "T_gpu_only": t_gpu_only,
        "T_hybrid": t_hybrid,
        "T_v": t_vanilla,
        "T_p": t_hybrid if t_hybrid is not None else t_gpu_only,
        "vanilla_per_variant_times_s": _per_variant_times(vanilla),
        "gpu_only_per_variant_times_s": _per_variant_times(gpu_only),
        "hybrid_per_variant_times_s": _per_variant_times(hybrid) if hybrid else None,
        "vanilla_session_total_s": vanilla.total_s,
        "gpu_only_session_total_s": gpu_only.total_s,
        "hybrid_session_total_s": hybrid.total_s if hybrid else None,
        "deployed_session_total_s": deployed_session_total_s,
        "session_speedup_vs_vanilla": session_speedup,
        "break_even_compile": break_even_compile,
        "compile_label": compile_label,
        "break_even_hybrid": break_even_hybrid,
        "hybrid_label": hybrid_label,
        "break_even_n": break_even_compile,
        "amortization_label": compile_label,
        "headline_surface_valid": deployed_surface_valid,
        "surface_validation": surface_validation,
        "surface_validations": surface_validation,
    }
    result = {
        "schema_version": _SCHEMA_VERSION,
        "implementation_status": "completed",
        "workload": workload_metadata(workload_name, config.sample_count),
        "consumes": {"rq3_config": a_config_name},
        "operating_point": operating_point,
        "rq2_selection": selection,
        "sessions": {
            "vanilla": _session_payload(vanilla),
            "gpu_only": _session_payload(gpu_only),
            "hybrid": _session_payload(hybrid) if hybrid else None,
        },
        "headline": {
            "session_speedup_vs_vanilla": session_speedup,
            "deployed_session_total_s": deployed_session_total_s,
            "operating_point": operating_point,
            "selected_policy": selected_policy,
            "compile_cold_start_s": compile_s,
            "calibration_s": cached_cell.calibration_s,
            "pool_startup_s": pool_startup_s,
            "break_even_compile": break_even_compile,
            "compile_label": compile_label,
            "break_even_hybrid": break_even_hybrid,
            "hybrid_label": hybrid_label,
        },
        "calibration": cached_cell.__dict__,
        "surface_validation": surface_validation,
        "surface_pairs": surface_pairs,
    }
    _progress(
        "complete",
        workload=workload_name,
        status=record["status"],
        session_speedup=session_speedup,
    )
    return Experiment3Result(
        status=record["status"],
        schema_version=_SCHEMA_VERSION,
        config=config,
        result=result,
        record=record,
    )


# --------------------------------------------------------------------------
#   Helpers
# --------------------------------------------------------------------------

def _skipped(
    config: Experiment3Config,
    experiment_1: Experiment1Result,
    reason: str,
    extra: dict[str, Any],
    workload_name: str | None = None,
) -> Experiment3Result:
    rq3_config = _resolve_rq3_config(experiment_1, workload_name) if workload_name else None
    record = {
        "status": "skipped",
        "implementation_status": "skipped",
        "workload": workload_name,
        "skip_reason": reason,
        "rq3_config": rq3_config,
        "session_speedup_vs_vanilla": None,
        "break_even_n": extra.get("break_even_n"),
        **extra,
    }
    result = {
        "schema_version": _SCHEMA_VERSION,
        "implementation_status": "skipped",
        "workload": workload_metadata(workload_name, config.sample_count) if workload_name else None,
        "consumes": {"rq3_config": rq3_config},
        "sessions": {"vanilla": None, "gpu_only": None, "hybrid": None},
        "headline": {
            "session_speedup_vs_vanilla": None,
            "break_even_compile": extra.get("break_even_compile"),
            "compile_label": extra.get("compile_label"),
            "break_even_hybrid": extra.get("break_even_hybrid"),
            "hybrid_label": extra.get("hybrid_label"),
        },
        "skip_reason": reason,
    }
    return Experiment3Result(
        status="skipped",
        schema_version=_SCHEMA_VERSION,
        config=config,
        result=result,
        record=record,
    )


def _calibration_starvation(
    config: Experiment3Config,
    experiment_1: Experiment1Result,
    experiment_2: Experiment2Result,
    *,
    workload_name: str,
    selection: dict[str, Any],
    rq3_config: str,
    point_chunk_size_k: int,
    probe_grid: GridSpec,
    cached_cell: calibration_mod.CalibratedCell,
) -> Experiment3Result:
    record = {
        "status": "calibration_starvation",
        "implementation_status": "completed",
        "workload": workload_name,
        "rq2_selection": selection,
        "rq3_config": rq3_config,
        "selected_policy": cached_cell.selected_policy,
        "selected_b_cell": cached_cell.__dict__,
        "calibration_s": cached_cell.calibration_s,
        "n_checkpoints": _N_CHECKPOINTS,
        "session_grid_resolution": config.session_grid.resolution,
        "calibration_grid_resolution": probe_grid.resolution,
        "point_chunk_size_K": point_chunk_size_k,
        "break_even_compile": math.inf,
        "compile_label": "refuted",
        "break_even_hybrid": math.inf,
        "hybrid_label": "refuted",
        "session_speedup_vs_vanilla": None,
        "source_exp_a_record": experiment_1.schema_version,
        "source_exp_b_record": experiment_2.schema_version,
    }
    result = {
        "schema_version": _SCHEMA_VERSION,
        "implementation_status": "completed",
        "workload": workload_metadata(workload_name, config.sample_count),
        "consumes": {"rq3_config": rq3_config},
        "operating_point": selection.get("operating_point"),
        "rq2_selection": selection,
        "sessions": {"vanilla": None, "gpu_only": None, "hybrid": None},
        "headline": {
            "session_speedup_vs_vanilla": None,
            "break_even_compile": math.inf,
            "compile_label": "refuted",
            "break_even_hybrid": math.inf,
            "hybrid_label": "refuted",
        },
        "calibration": cached_cell.__dict__,
    }
    return Experiment3Result(
        status="calibration_starvation",
        schema_version=_SCHEMA_VERSION,
        config=config,
        result=result,
        record=record,
    )


_THRESHOLD_REACHED = {"crosses_within_range", "wins_at_native", "non_monotone"}


def _select_from_experiment_2(experiment_2: Experiment2Result) -> dict[str, Any] | None:
    """Select the workload whose hybrid threshold is reached within the swept
    ladder and run RQ3 at that threshold operating point (the threshold rung's
    slowdown). Prefer a native threshold (slowdown 1.0), where the win is a
    practical-hardware claim; fall back to a slowed threshold, a controlled
    demonstration. Among candidates take the lowest threshold slowdown (hybrid
    pays off earliest), breaking ties by the strongest win at that rung.
    Returns None when no workload's threshold is reached, which means
    calibration cannot pay off anywhere in the explored range."""
    workloads = (experiment_2.record or {}).get("workloads") or {}
    eligible: list[dict[str, Any]] = []
    for workload_name, payload in workloads.items():
        if not isinstance(payload, dict) or payload.get("status") != "completed":
            continue
        if payload.get("threshold_status") not in _THRESHOLD_REACHED:
            continue
        threshold_slowdown = payload.get("threshold_slowdown")
        if threshold_slowdown is None:
            continue
        threshold_rung = _rung_at(payload.get("ladder") or [], threshold_slowdown)
        predictor = payload.get("regime_predictor") or {}
        eligible.append(
            {
                "workload_name": workload_name,
                "threshold_status": payload.get("threshold_status"),
                "slowdown_factor": float(threshold_slowdown),
                "achieved_ratio_at_threshold": payload.get("achieved_ratio_at_threshold"),
                "speedup_ci_low": (threshold_rung or {}).get("speedup_ci_low"),
                "r_native": predictor.get("r_native"),
            }
        )
    if not eligible:
        return None

    def _key(item: dict[str, Any]) -> tuple[float, float]:
        ci_low = item["speedup_ci_low"]
        ci_low = ci_low if isinstance(ci_low, (int, float)) else float("-inf")
        return (item["slowdown_factor"], -ci_low)

    best = sorted(eligible, key=_key)[0]
    best["operating_point"] = "native" if best["slowdown_factor"] == 1.0 else "slowed"
    return best


def _rung_at(ladder: list[dict[str, Any]], slowdown: float) -> dict[str, Any] | None:
    for rung in ladder:
        if isinstance(rung, dict) and rung.get("slowdown_factor") == slowdown:
            return rung
    return None


def _break_even(one_time_s: float, per_variant_saving_s: float) -> int | float:
    if per_variant_saving_s <= 0:
        return math.inf
    return int(math.ceil(one_time_s / per_variant_saving_s))


def _amortization_label(break_even: int | float) -> str:
    if math.isinf(float(break_even)):
        return "refuted"
    if break_even <= _N_CHECKPOINTS:
        return "supported"
    return "asymptotic_only"


def _progress(event: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"[exp_3] {event}{suffix}", flush=True)


def _resolve_rq3_config(experiment_1: Experiment1Result, workload_name: str) -> str:
    by_workload = experiment_1.record.get("rq3_config_by_workload") or {}
    return by_workload.get(workload_name) or experiment_1.rq3_config or "baseline"


def _measure_compile_cost(
    a_config: GpuCandidate,
    task: Any,
    grid: GridSpec,
    *,
    gpu_batch_size: int,
    device: torch.device,
    seed: int,
    gpu_slowdown_factor: float,
) -> float:
    """Compile cold-start (s) for the A config, measured once at the session GPU
    operating point. The cold-start is the torch.compile graph-build time, which
    depends on the model and chunk shape but not on grid size, so a small probe
    grid reproduces the session's compile. Returns 0.0 for A configs that do not
    compile, so the one-time setup cost stays uniform across workloads."""
    if a_config.role not in _COMPILING_ROLES:
        return 0.0
    output = run_standalone(
        a_config, task, grid,
        batch_size=gpu_batch_size, device=device, seed=seed,
        gpu_slowdown_factor=gpu_slowdown_factor,
    )
    return float(output.compile_cold_start_s or 0.0)


def _parse_gpu_candidate(name: str) -> GpuCandidate:
    if name == "baseline":
        return GpuCandidate.baseline()
    if name == "compiled":
        return GpuCandidate.compiled()
    match = _CANDIDATE_K_RE.search(name)
    if name.startswith("vmapped_k") and match:
        return GpuCandidate.vmapped(int(match.group(1)))
    if name.startswith("compiled_vmapped_k") and match:
        return GpuCandidate.compiled_vmapped(int(match.group(1)))
    raise ValueError(f"unrecognized GpuCandidate name: {name!r}")


def _point_chunk_size_for(candidate: GpuCandidate) -> int:
    if candidate.role in ("vmapped", "compiled_vmapped"):
        return int(candidate.point_chunk_size or 1)
    return 1


def _probe_grid_for(p_max: int, point_chunk_size_k: int) -> GridSpec:
    """Smallest m where m² >= K + 4*p_max per calibration-cache-plan."""
    required_points = int(point_chunk_size_k) + (4 * int(p_max))
    m = int(math.ceil(math.sqrt(required_points)))
    return GridSpec(resolution=m, scale=1.0)


def _validate_session_surfaces(
    candidate: sessions_mod.SessionRecord,
    vanilla: sessions_mod.SessionRecord,
    *,
    arm: str,
    surface_gate,
) -> list[dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    for candidate_item, vanilla_item in zip(candidate.per_checkpoint, vanilla.per_checkpoint):
        try:
            validation = validate_surface(
                candidate_item.records, vanilla_item.records, surface_gate,
            )
        except AssertionError as exc:
            validation = {
                "point_count": None,
                "mismatch_count": None,
                "valid": False,
                "rel_tol": surface_gate.rel_tol,
                "abs_tol": surface_gate.abs_tol,
                "max_abs_error": None,
                "rmse": None,
                "first_mismatches": [],
                "error": str(exc),
            }
        validations.append(
            {
                "arm": arm,
                "checkpoint_path": candidate_item.checkpoint_path,
                **validation,
            }
        )
    return validations


def _reference_surface_validations(
    session: sessions_mod.SessionRecord,
) -> list[dict[str, Any]]:
    return [
        {
            "arm": "vanilla",
            "checkpoint_path": item.checkpoint_path,
            "point_count": len(item.records),
            "mismatch_count": 0,
            "valid": True,
            "reference": True,
        }
        for item in session.per_checkpoint
    ]


def _surface_valid(validations: list[dict[str, Any]] | None) -> bool:
    return bool(validations) and all(bool(item.get("valid")) for item in validations)


def _session_surface_pairs(
    candidate: sessions_mod.SessionRecord,
    vanilla: sessions_mod.SessionRecord,
    *,
    arm: str,
) -> list[dict[str, Any]]:
    return [
        {
            "baseline": "vanilla",
            "candidate": arm,
            "checkpoint_path": candidate_item.checkpoint_path,
            "baseline_records": vanilla_item.records,
            "candidate_records": candidate_item.records,
        }
        for candidate_item, vanilla_item in zip(candidate.per_checkpoint, vanilla.per_checkpoint)
    ]


def _per_variant_times(session: sessions_mod.SessionRecord) -> list[float]:
    return [item.t_grid_s for item in session.per_checkpoint]


def _session_payload(session: sessions_mod.SessionRecord) -> dict[str, Any]:
    return {
        "per_checkpoint_total_s": [item.t_grid_s for item in session.per_checkpoint],
        "checkpoint_paths": [item.checkpoint_path for item in session.per_checkpoint],
        "sum_per_checkpoint_s": session.sum_per_checkpoint_s,
        "total_s": session.total_s,
        "mean_t_grid_s": session.mean_t_grid_s,
        "sigma_rel": session.sigma_rel,
        "diagnostics": [item.diagnostics for item in session.per_checkpoint],
    }
