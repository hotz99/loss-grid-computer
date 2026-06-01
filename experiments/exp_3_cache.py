from __future__ import annotations

import math
import re
from dataclasses import replace as _replace
from typing import Any

import torch

from experiments import composition_selection as selection_mod
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
from experiments.workloads import WORKLOADS, task_for_workload


_SCHEMA_VERSION = "experiment-3-composition-v1"
_N_CHECKPOINTS = 4
_CANDIDATE_K_RE = re.compile(r"_k(\d+)$")
_COMPILING_ROLES = {"compiled", "compiled_vmapped"}
# The composition sweep runs at native r. The RQ2 slowdown instrument is an
# optional boundary overlay, not part of the default sweep, so the GPU path is
# never slowed here.
_NATIVE_SLOWDOWN = 1.0


def run(
    config: Experiment3Config,
    experiment_1: Experiment1Result,
    experiment_2: Experiment2Result,
) -> Experiment3Result:
    """Cross-axis composition sweep (RQ3): per (workload, platform) cell, compose
    the RQ1-selected GPU evaluator with the native selection probe and read the
    composition verdict q_cross = T_gpu_only / T_hybrid. The full workload set is
    swept with no affinity filter, in the same shape as RQ1/RQ2. exp2 is consumed
    only for r_native per workload, not for a selection."""
    _progress("start")
    device = device_mod.resolve(config.device)
    workload_names = _sweep_workloads(experiment_1, experiment_2)
    _progress("sweep", device=device.type, workloads=len(workload_names))

    cells: dict[str, dict[str, Any]] = {}
    surface_pairs: list[dict[str, Any]] = []
    for workload_name in workload_names:
        cell_record, cell_pairs = _run_cell(
            config, experiment_1, experiment_2, workload_name, device,
        )
        cells[workload_name] = cell_record
        surface_pairs.extend(cell_pairs)

    record = {
        "status": "completed",
        "implementation_status": "completed",
        "device": device.type,
        "n_checkpoints": _N_CHECKPOINTS,
        "session_grid_resolution": config.session_grid.resolution,
        "source_exp_a_record": experiment_1.schema_version,
        "source_exp_b_record": experiment_2.schema_version,
        "rq3_config_by_workload": {
            name: cell.get("cell", {}).get("rq3_config") for name, cell in cells.items()
        },
        "cells": cells,
    }
    result = {
        "schema_version": _SCHEMA_VERSION,
        "implementation_status": "completed",
        "device": device.type,
        "cells": cells,
        "surface_pairs": surface_pairs,
    }
    _progress(
        "complete",
        workloads=len(cells),
        completed=sum(1 for c in cells.values() if c["status"] == "completed"),
    )
    return Experiment3Result(
        status="completed",
        schema_version=_SCHEMA_VERSION,
        config=config,
        result=result,
        record=record,
    )


# --------------------------------------------------------------------------
#   Per-cell composition sweep
# --------------------------------------------------------------------------

def _run_cell(
    config: Experiment3Config,
    experiment_1: Experiment1Result,
    experiment_2: Experiment2Result,
    workload_name: str,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    r_native = _r_native_for(experiment_2, workload_name)
    a_config_name = _resolve_rq3_config(experiment_1, workload_name)
    if workload_name not in WORKLOADS:
        _progress("skip", workload=workload_name, reason="unknown_workload")
        return _skipped_cell(
            workload_name, device, "unknown_workload",
            rq3_config=a_config_name, r_native=r_native, config=config,
        ), []

    a_config = _parse_gpu_candidate(a_config_name)
    point_chunk_size_k = _point_chunk_size_for(a_config)
    _progress(
        "cell", workload=workload_name, platform=device.type,
        rq3_config=a_config_name, point_chunk_size_K=point_chunk_size_k,
    )

    task = task_for_workload(workload_name, config.sample_count)
    try:
        checkpoints = list_same_family_checkpoints(task, _N_CHECKPOINTS)
    except FileNotFoundError as exc:
        _progress("skip", workload=workload_name, reason="missing_checkpoints")
        return _skipped_cell(
            workload_name, device, "missing_checkpoints",
            rq3_config=a_config_name, r_native=r_native, config=config,
            extra={"error": str(exc)},
        ), []
    _progress("checkpoints", workload=workload_name, count=len(checkpoints))

    workers = selection_mod.cpu_worker_candidates(config.max_cpu_worker_candidate)
    probe_grid = _probe_grid_for(max(workers), point_chunk_size_k)
    task_at_ckpt0 = _replace(task, checkpoint_path=checkpoints[0])

    _progress(
        "selection_probe_baseline",
        workload=workload_name,
        grid_resolution=probe_grid.resolution,
    )
    selection_probe_baseline = run_standalone(
        a_config, task_at_ckpt0, probe_grid,
        batch_size=config.gpu_batch_size, device=device, seed=config.seed,
        gpu_slowdown_factor=_NATIVE_SLOWDOWN,
    )
    batches = selection_mod.cpu_batch_size_candidates(
        task.dataset.sample_count, config.gpu_batch_size,
    )
    _progress(
        "composition_selection",
        workload=workload_name,
        cpu_worker_candidates=len(workers),
    )
    selection = selection_mod.select_composition(
        task_at_ckpt0, probe_grid,
        gpu_batch_size=config.gpu_batch_size,
        baseline_total_s=selection_probe_baseline.total_grid_s,
        cpu_workers=workers, cpu_batch_sizes=batches,
        patience=config.composition_selection_patience,
        device=device, seed=config.seed,
        gpu_slowdown_factor=_NATIVE_SLOWDOWN,
        gpu_candidate=a_config,
    )
    if selection.max_hybrid_cpu_points <= 0:
        _progress("selection_starvation", workload=workload_name)
        return _starved_cell(
            workload_name, device, a_config_name, a_config, point_chunk_size_k,
            probe_grid, selection, r_native, config,
        ), []

    compile_s = _measure_compile_cost(
        a_config, task_at_ckpt0, probe_grid,
        gpu_batch_size=config.gpu_batch_size, device=device, seed=config.seed,
    )
    _progress("compile_cost", workload=workload_name, compile_s=compile_s)

    _progress("vanilla_session", workload=workload_name, checkpoints=len(checkpoints))
    vanilla = sessions_mod.vanilla_session(
        task, config.session_grid, checkpoints,
        batch_size=config.gpu_batch_size, device=device, seed=config.seed,
        gpu_slowdown_factor=_NATIVE_SLOWDOWN,
    )
    _progress("gpu_only_session", workload=workload_name, checkpoints=len(checkpoints))
    gpu_only = sessions_mod.gpu_only_session(
        task, config.session_grid, checkpoints,
        gpu_candidate=a_config, batch_size=config.gpu_batch_size,
        device=device, seed=config.seed, gpu_slowdown_factor=_NATIVE_SLOWDOWN,
    )
    hybrid = None
    pool_startup_s = 0.0
    if selection.selected_path == "gpu_cpu_hybrid":
        _progress("hybrid_pool_session", workload=workload_name, checkpoints=len(checkpoints))
        hybrid, pool_startup_s = sessions_mod.hybrid_pool_session(
            task, config.session_grid, checkpoints,
            selection=selection, gpu_candidate=a_config, device=device, seed=config.seed,
            gpu_slowdown_factor=_NATIVE_SLOWDOWN,
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

    # Composition verdict. A failing surface on a measured timing arm suppresses
    # the verdict (the timing claim is unsupported), per the canon correctness
    # gate. gpu_only is itself the verdict: A dominates B, q_cross = 1.
    arms_surface_valid = gpu_only_surface_valid and hybrid_surface_valid
    if not arms_surface_valid:
        q_cross = None
        composition_verdict = "surface_invalid"
    elif selection.selected_path == "gpu_only" or t_hybrid is None:
        q_cross = 1.0
        composition_verdict = "dominate"
    else:
        q_cross = t_gpu_only / t_hybrid if t_hybrid > 0 else None
        composition_verdict = _composition_verdict(q_cross)

    rq3_config_compiles = a_config.role in _COMPILING_ROLES
    n_star_compile = _n_star_compile(
        compile_s, t_vanilla, t_gpu_only, rq3_config_compiles
    )
    compile_reuse_label = _compile_reuse_label(n_star_compile, rq3_config_compiles)
    session_speedup = _session_speedup(
        t_vanilla, t_gpu_only, compile_s,
        valid=gpu_only_surface_valid,
    )

    cell_pairs = _session_surface_pairs(
        gpu_only, vanilla, arm="gpu_only", workload=workload_name
    )
    if hybrid is not None:
        cell_pairs.extend(
            _session_surface_pairs(hybrid, vanilla, arm="hybrid", workload=workload_name)
        )

    record = {
        "status": "completed",
        "implementation_status": "completed",
        "cell": {
            "workload": workload_name,
            "platform": device.type,
            "r_native": r_native,
            "cpu_pool_size": selection.cpu_workers or 0,
            "source_exp_a_record": experiment_1.schema_version,
            "rq3_config": a_config_name,
        },
        "n_checkpoints": _N_CHECKPOINTS,
        "session_grid_resolution": config.session_grid.resolution,
        "selection_probe_grid_resolution": probe_grid.resolution,
        "point_chunk_size_K": point_chunk_size_k,
        "selected_path": selection.selected_path,
        "composition_selection": selection.__dict__,
        "q_cross": q_cross,
        "composition_verdict": composition_verdict,
        "selection_probe_s": selection.selection_probe_s,
        "compile_s": compile_s,
        "pool_startup_s": pool_startup_s,
        "rq3_config_compiles": rq3_config_compiles,
        "T_vanilla": t_vanilla,
        "vanilla_per_variant_times_s": _per_variant_times(vanilla),
        "T_gpu_only": t_gpu_only,
        "gpu_only_per_variant_times_s": _per_variant_times(gpu_only),
        "T_hybrid": t_hybrid,
        "hybrid_per_variant_times_s": _per_variant_times(hybrid) if hybrid else None,
        "N_star_compile": _n_star_json(n_star_compile),
        "compile_reuse_label": compile_reuse_label,
        "session_speedup_vs_vanilla": session_speedup,
        "surface_validation": surface_validation,
    }
    _progress(
        "cell_complete", workload=workload_name,
        verdict=composition_verdict, q_cross=q_cross,
    )
    return record, cell_pairs


# --------------------------------------------------------------------------
#   Cell terminals (skip / starvation)
# --------------------------------------------------------------------------

def _skipped_cell(
    workload_name: str,
    device: torch.device,
    reason: str,
    *,
    rq3_config: str | None,
    r_native: float | None,
    config: Experiment3Config,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "implementation_status": "skipped",
        "skip_reason": reason,
        "cell": {
            "workload": workload_name,
            "platform": device.type,
            "r_native": r_native,
            "cpu_pool_size": 0,
            "rq3_config": rq3_config,
        },
        "session_grid_resolution": config.session_grid.resolution,
        "selected_path": None,
        "q_cross": None,
        "composition_verdict": "skipped",
        "session_speedup_vs_vanilla": None,
        "N_star_compile": None,
        "compile_reuse_label": "undefined",
        **(extra or {}),
    }


def _starved_cell(
    workload_name: str,
    device: torch.device,
    rq3_config: str,
    a_config: GpuCandidate,
    point_chunk_size_k: int,
    probe_grid: GridSpec,
    selection: selection_mod.CompositionSelection,
    r_native: float | None,
    config: Experiment3Config,
) -> dict[str, Any]:
    return {
        "status": "selection_starvation",
        "implementation_status": "completed",
        "skip_reason": "selection_starvation",
        "cell": {
            "workload": workload_name,
            "platform": device.type,
            "r_native": r_native,
            "cpu_pool_size": selection.cpu_workers or 0,
            "rq3_config": rq3_config,
        },
        "n_checkpoints": _N_CHECKPOINTS,
        "session_grid_resolution": config.session_grid.resolution,
        "selection_probe_grid_resolution": probe_grid.resolution,
        "point_chunk_size_K": point_chunk_size_k,
        "selected_path": selection.selected_path,
        "composition_selection": selection.__dict__,
        "q_cross": None,
        "composition_verdict": "selection_starvation",
        "selection_probe_s": selection.selection_probe_s,
        "compile_s": None,
        "pool_startup_s": None,
        "rq3_config_compiles": a_config.role in _COMPILING_ROLES,
        "N_star_compile": None,
        "compile_reuse_label": "undefined",
        "session_speedup_vs_vanilla": None,
    }


# --------------------------------------------------------------------------
#   Verdict / compiled evaluator reuse helpers
# --------------------------------------------------------------------------

def _composition_verdict(q_cross: float | None) -> str:
    if q_cross is None:
        return "surface_invalid"
    return "complement" if q_cross > 1.0 else "dominate"


def _n_star_compile(
    compile_s: float,
    t_vanilla: float,
    t_gpu_only: float,
    compiles: bool,
) -> int | float | None:
    """N*_compile = ceil(compile_s / (T_vanilla - T_gpu_only)).

    Undefined (None) when RQ1 selected the baseline: no compile cost to recover.
    Infinite when the compiled path is not faster per variant than vanilla."""
    if not compiles:
        return None
    saving = t_vanilla - t_gpu_only
    if saving <= 0:
        return math.inf
    return int(math.ceil(compile_s / saving))


def _compile_reuse_label(n_star: int | float | None, compiles: bool) -> str:
    if not compiles or n_star is None:
        return "undefined"
    if math.isinf(float(n_star)):
        return "refuted"
    if n_star <= _N_CHECKPOINTS:
        return "supported"
    return "supported_asymptotically"


def _session_speedup(
    t_vanilla: float,
    t_gpu_only: float,
    compile_s: float,
    *,
    valid: bool,
) -> float | None:
    """Honest deployed-session headline at N = 4: (N*T_vanilla) charged against
    the compile cold-start the deployed stack actually paid plus N warm GPU-only
    grids. Suppressed when the GPU-only surface is invalid."""
    denominator = compile_s + (_N_CHECKPOINTS * t_gpu_only)
    if not valid or denominator <= 0:
        return None
    return (_N_CHECKPOINTS * t_vanilla) / denominator


def _n_star_json(n_star: int | float | None) -> int | str | None:
    if n_star is None:
        return None
    if math.isinf(float(n_star)):
        return "infinite"
    return int(n_star)


# --------------------------------------------------------------------------
#   Sweep inputs
# --------------------------------------------------------------------------

def _sweep_workloads(
    experiment_1: Experiment1Result,
    experiment_2: Experiment2Result,
) -> tuple[str, ...]:
    """Workload set for the sweep: the cells RQ1 reports a winner for. Falls back
    to the exp2 workload set when the RQ1 per-workload map is absent."""
    by_workload = experiment_1.record.get("rq3_config_by_workload") or {}
    if by_workload:
        return tuple(by_workload.keys())
    workloads = (experiment_2.record or {}).get("workloads") or {}
    return tuple(workloads.keys())


def _r_native_for(experiment_2: Experiment2Result, workload_name: str) -> float | None:
    workloads = (experiment_2.record or {}).get("workloads") or {}
    payload = workloads.get(workload_name)
    if not isinstance(payload, dict):
        return None
    return (payload.get("regime_predictor") or {}).get("r_native")


def _resolve_rq3_config(experiment_1: Experiment1Result, workload_name: str) -> str:
    by_workload = experiment_1.record.get("rq3_config_by_workload") or {}
    return by_workload.get(workload_name) or experiment_1.rq3_config or "baseline"


# --------------------------------------------------------------------------
#   GPU config + probe grid
# --------------------------------------------------------------------------

def _measure_compile_cost(
    a_config: GpuCandidate,
    task: Any,
    grid: GridSpec,
    *,
    gpu_batch_size: int,
    device: torch.device,
    seed: int,
) -> float:
    """Compile cold-start (s) for the rq3_config, measured once at native r. The
    cold-start is the torch.compile graph-build time, which depends on the model
    and chunk shape but not on grid size, so a small probe grid reproduces the
    session's compile. Returns 0.0 for configs that do not compile."""
    if a_config.role not in _COMPILING_ROLES:
        return 0.0
    output = run_standalone(
        a_config, task, grid,
        batch_size=gpu_batch_size, device=device, seed=seed,
        gpu_slowdown_factor=_NATIVE_SLOWDOWN,
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
    """Smallest m where m^2 >= K + 4*p_max for the selection probe."""
    required_points = int(point_chunk_size_k) + (4 * int(p_max))
    m = int(math.ceil(math.sqrt(required_points)))
    return GridSpec(resolution=m, scale=1.0)


# --------------------------------------------------------------------------
#   Surface validation + serialization
# --------------------------------------------------------------------------

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
    workload: str,
) -> list[dict[str, Any]]:
    return [
        {
            "workload": workload,
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


def _progress(event: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"[exp_3] {event}{suffix}", flush=True)
