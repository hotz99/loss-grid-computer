from __future__ import annotations

import math
import re
from dataclasses import replace as _replace
from typing import Any

from experiments import calibration as calibration_mod
from experiments import device as device_mod
from experiments import sessions as sessions_mod
from experiments.candidates import GpuCandidate
from experiments.candidates import baseline as baseline_candidate
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
_PATIENCE = 3
_CANDIDATE_K_RE = re.compile(r"_k(\d+)$")


def run(
    config: Experiment3Config,
    experiment_1: Experiment1Result,
    experiment_2: Experiment2Result,
) -> Experiment3Result:
    """Calibrate-once-cache for the hetero scheduler on top of A's RQ1 winner
    (canonical-overview.md L231-243). RQ3 filters RQ2 for a hybrid-affinity
    (workload, regime), inherits that regime's slowdown as a uniform operating
    point, and measures whether the one-time calibration cost amortizes over
    the session. Bounded calibration discipline [ansel2014opentuner]; the
    headline is cumulative T_session speedup at the LossLens-scoped session
    size, with break-even N as context."""
    _progress("start")
    selection = _select_from_experiment_2(experiment_2)
    if selection is None:
        _progress("skip", reason="no_hybrid_affinity")
        return _skipped(
            config, experiment_1, "no_hybrid_affinity",
            {"amortization_label": "refuted", "rq2_selection": None},
        )
    workload_name = selection["workload_name"]
    slowdown = selection["slowdown_factor"]
    operating_point = selection["operating_point"]
    _progress(
        "selected", workload=workload_name, regime=selection["regime"],
        slowdown=slowdown, operating_point=operating_point,
    )
    if workload_name not in WORKLOADS:
        _progress("skip", workload=workload_name, reason="unknown_workload")
        return _skipped(
            config, experiment_1, "unknown_workload", {}, workload_name=workload_name,
        )

    a_config_name = _resolve_rq3_config(experiment_1, workload_name)
    a_config = _parse_gpu_candidate(a_config_name)
    _progress("a_config", workload=workload_name, candidate=a_config_name)

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
    probe_grid = _probe_grid_for(max(workers))

    task_at_ckpt0 = _replace(task, checkpoint_path=checkpoints[0])
    _progress("baseline_seed", workload=workload_name, grid_resolution=probe_grid.resolution)
    baseline_seed = baseline_candidate.run(
        task_at_ckpt0, probe_grid,
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
        baseline_total_s=baseline_seed.total_grid_s,
        cpu_workers=workers, cpu_batch_sizes=batches,
        patience=_PATIENCE,
        device=device, seed=config.seed,
        gpu_slowdown_factor=slowdown,
        gpu_candidate=a_config,
    )

    _progress("cached_composed_session", workload=workload_name, checkpoints=len(checkpoints))
    composed = sessions_mod.cached_composed_session(
        task, config.session_grid, checkpoints,
        cell=cached_cell, gpu_candidate=a_config, device=device, seed=config.seed,
        gpu_slowdown_factor=slowdown,
    )
    _progress("vanilla_session", workload=workload_name, checkpoints=len(checkpoints))
    vanilla = sessions_mod.vanilla_session(
        task, config.session_grid, checkpoints,
        batch_size=config.gpu_batch_size, device=device, seed=config.seed,
        gpu_slowdown_factor=slowdown,
    )

    _progress("validate", workload=workload_name)
    surface_validations = _validate_session_surfaces(composed, vanilla)
    surface_valid = all(item["valid"] for item in surface_validations)

    session_speedup = (
        vanilla.total_s / composed.total_s if composed.total_s > 0 else None
    )
    break_even = sessions_mod.break_even_n(
        vanilla.mean_t_grid_s, composed.mean_t_grid_s, cached_cell.calibration_s,
    )
    amortization_label = _amortization_label(session_speedup, break_even, surface_valid)

    record = {
        "status": "completed" if surface_valid else "completed_with_invalid_surface",
        "implementation_status": "completed",
        "device": device.type,
        "workload": workload_name,
        "operating_point": operating_point,
        "slowdown_factor": slowdown,
        "rq2_selection": selection,
        "rq3_config": a_config_name,
        "selected_b_cell": cached_cell.__dict__,
        "calibration_s": cached_cell.calibration_s,
        "n_checkpoints": _N_CHECKPOINTS,
        "session_grid_resolution": config.session_grid.resolution,
        "calibration_grid_resolution": probe_grid.resolution,
        "T_v": vanilla.mean_t_grid_s,
        "T_v_sigma_rel": vanilla.sigma_rel,
        "T_p": composed.mean_t_grid_s,
        "T_p_sigma_rel": composed.sigma_rel,
        "vanilla_session_total_s": vanilla.total_s,
        "cached_composed_session_total_s": composed.total_s,
        "session_speedup_vs_vanilla": session_speedup,
        "break_even_n": break_even,
        "break_even_meets_lossLens_target": (
            break_even is not None and break_even <= _N_CHECKPOINTS
        ),
        "amortization_label": amortization_label,
        "surface_validations": surface_validations,
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
            "cached_composed": _session_payload(composed),
        },
        "headline": {
            "session_speedup_vs_vanilla": session_speedup,
            "break_even_n": break_even,
            "operating_point": operating_point,
            "amortization_label": amortization_label,
        },
        "calibration": cached_cell.__dict__,
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
        "sessions": {"vanilla": None, "cached_composed": None},
        "headline": {
            "session_speedup_vs_vanilla": None,
            "break_even_n": record.get("break_even_n"),
            "amortization_label": extra.get("amortization_label"),
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


def _select_from_experiment_2(experiment_2: Experiment2Result) -> dict[str, Any] | None:
    """Filter RQ2 for a hybrid-affinity (workload, regime): a `hybrid_wins`
    verdict. Prefer a native regime (slowdown 1.0), where the win is a
    practical-hardware claim; fall back to a slowed regime, where the win is a
    controlled demonstration. Among candidates take the strongest supported win
    (highest speedup CI low bound). Returns None when no regime has hybrid
    affinity, which means calibration cannot pay off anywhere."""
    workloads = (experiment_2.record or {}).get("workloads") or {}
    eligible: list[dict[str, Any]] = []
    for workload_name, payload in workloads.items():
        if not isinstance(payload, dict) or payload.get("status") != "completed":
            continue
        regimes = payload.get("regimes") or {}
        for regime_name, regime in regimes.items():
            if not isinstance(regime, dict) or regime.get("claim_status") != "hybrid_wins":
                continue
            eligible.append(
                {
                    "workload_name": workload_name,
                    "regime": regime_name,
                    "slowdown_factor": float(regime.get("slowdown_factor") or 1.0),
                    "speedup_ci_low": regime.get("speedup_ci_low"),
                }
            )
    if not eligible:
        return None

    def _key(item: dict[str, Any]) -> tuple[int, float]:
        is_slowed = 0 if item["regime"] == "native" else 1
        ci_low = item["speedup_ci_low"]
        ci_low = ci_low if isinstance(ci_low, (int, float)) else float("-inf")
        return (is_slowed, -ci_low)

    best = sorted(eligible, key=_key)[0]
    best["operating_point"] = "native" if best["slowdown_factor"] == 1.0 else "slowed"
    return best


def _amortization_label(
    session_speedup: float | None,
    break_even: int | None,
    surface_valid: bool,
) -> str:
    """Verdict taxonomy per methods-calibration: surface failure is
    inconclusive; speedup <= 1 or no per-checkpoint saving is refuted;
    otherwise supported when break-even is within the session size, else
    supported asymptotically."""
    if not surface_valid or session_speedup is None:
        return "inconclusive"
    if session_speedup <= 1.0 or break_even is None:
        return "refuted"
    if break_even <= _N_CHECKPOINTS:
        return "supported"
    return "supported_asymptotically"


def _progress(event: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"[exp_3] {event}{suffix}", flush=True)


def _resolve_rq3_config(experiment_1: Experiment1Result, workload_name: str) -> str:
    by_workload = experiment_1.record.get("rq3_config_by_workload") or {}
    return by_workload.get(workload_name) or experiment_1.rq3_config or "baseline"


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


def _probe_grid_for(p_max: int) -> GridSpec:
    """Smallest m where m² ≥ 4 × (1 + p_max) per calibration-cache-plan."""
    required_points = 4 * (1 + int(p_max))
    m = int(math.ceil(math.sqrt(required_points)))
    return GridSpec(resolution=m, scale=1.0)


def _validate_session_surfaces(
    composed: sessions_mod.SessionRecord,
    vanilla: sessions_mod.SessionRecord,
) -> list[dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    for composed_item, vanilla_item in zip(composed.per_checkpoint, vanilla.per_checkpoint):
        validation = validate_surface(composed_item.records, vanilla_item.records)
        validations.append(
            {
                "checkpoint_path": composed_item.checkpoint_path,
                **validation,
            }
        )
    return validations


def _session_payload(session: sessions_mod.SessionRecord) -> dict[str, Any]:
    return {
        "per_checkpoint_total_s": [item.t_grid_s for item in session.per_checkpoint],
        "checkpoint_paths": [item.checkpoint_path for item in session.per_checkpoint],
        "sum_per_checkpoint_s": session.sum_per_checkpoint_s,
        "total_s": session.total_s,
        "mean_t_grid_s": session.mean_t_grid_s,
        "sigma_rel": session.sigma_rel,
    }
