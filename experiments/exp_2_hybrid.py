from __future__ import annotations

from typing import Any

import torch

from experiments import calibration as calibration_mod
from experiments import device as device_mod
from experiments.candidates import baseline as baseline_candidate
from experiments.candidates import hybrid as hybrid_candidate
from experiments.probes import throughput as throughput_probe
from experiments.schemas import (
    Experiment2Config,
    Experiment2Result,
    GridSpec,
    TrialSpec,
)
from experiments.stats import paired_speedups, speedup_claim_status
from experiments.surface_gate import validate_surface
from experiments.workloads import WORKLOADS, task_for_workload, workload_metadata


_SCHEMA_VERSION = "experiment-2-hybrid-v1"
# r_native is a per-point throughput ratio that only sets where the slowdown
# lands, so a small probe grid suffices and an exact parity point is unneeded.
_PROBE_GRID_RESOLUTION = 4


def plan(config: Experiment2Config) -> tuple[TrialSpec, ...]:
    trials: list[TrialSpec] = []
    for workload_name in config.workload_names:
        for repeat in range(config.repeats):
            order = (
                ("vanilla", "hybrid")
                if repeat % 2 == 0
                else ("hybrid", "vanilla")
            )
            for candidate in order:
                trials.append(
                    TrialSpec(
                        experiment="B",
                        workload_name=workload_name,
                        candidate=candidate,
                        repeat=repeat,
                        trial_order=order,
                        control={"regime": "native_and_parity_probe"},
                    )
                )
    return tuple(trials)


def run(config: Experiment2Config) -> Experiment2Result:
    _progress("start", workload_count=len(config.workload_names), repeats=config.repeats)
    device = device_mod.resolve(config.device)
    trials = plan(config)
    _progress("planned", device=device.type, trial_count=len(trials))
    workloads_payload: dict[str, Any] = {}
    for workload_name in config.workload_names:
        _progress("workload", workload=workload_name)
        if workload_name not in WORKLOADS:
            workloads_payload[workload_name] = {
                "status": "unknown_workload",
                "workload": workload_metadata(workload_name, config.sample_count),
            }
            _progress("workload_complete", workload=workload_name, status="unknown_workload")
            continue
        task = task_for_workload(workload_name, config.sample_count)
        workloads_payload[workload_name] = _run_workload(task, config, device, workload_name)
        _progress(
            "workload_complete",
            workload=workload_name,
            status=workloads_payload[workload_name]["status"],
        )

    record = {
        "status": "completed",
        "implementation_status": "completed",
        "device": device.type,
        "trial_count": len(trials),
        "workloads": workloads_payload,
    }
    _progress("complete", workload_count=len(workloads_payload))
    result = {
        "schema_version": _SCHEMA_VERSION,
        "implementation_status": "completed",
        "trials": trials,
        "workloads": workloads_payload,
    }
    return Experiment2Result(
        status="completed",
        schema_version=_SCHEMA_VERSION,
        config=config,
        result=result,
        record=record,
    )


# --------------------------------------------------------------------------
#   Per-workload pipeline
# --------------------------------------------------------------------------

def _run_workload(
    task,
    config: Experiment2Config,
    device: torch.device,
    workload_name: str,
) -> dict[str, Any]:
    _progress("probe", workload=workload_name)
    cpu_batch_for_probe = max(1, min(config.gpu_batch_size, task.dataset.sample_count))
    probe = throughput_probe.measure(
        task, GridSpec(_PROBE_GRID_RESOLUTION, config.grid.scale),
        gpu_batch_size=config.gpu_batch_size,
        cpu_batch_size=cpu_batch_for_probe,
        gpu_device=device, seed=config.seed,
    )
    regimes: list[tuple[str, float]] = [("native", 1.0)]
    if probe.r_native < 1.0:
        regimes.append(("parity", probe.slowdown_for_parity()))

    per_regime: dict[str, Any] = {}
    for regime_name, slowdown in regimes:
        _progress("regime", workload=workload_name, regime=regime_name, slowdown=slowdown)
        per_regime[regime_name] = _run_regime(
            task, config, device, slowdown, probe.r_native, regime_name,
        )

    return {
        "status": "completed",
        "workload": workload_metadata(workload_name, config.sample_count),
        "regime_predictor": {
            "cpu_throughput_pts_s": probe.cpu_throughput_pts_s,
            "gpu_throughput_pts_s": probe.gpu_throughput_pts_s,
            "r_native": probe.r_native,
            "cpu_total_s": probe.cpu_total_s,
            "gpu_total_s": probe.gpu_total_s,
        },
        "regimes": per_regime,
    }


def _progress(event: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"[exp_2] {event}{suffix}", flush=True)


def _run_regime(
    task,
    config: Experiment2Config,
    device: torch.device,
    slowdown: float,
    r_native: float,
    regime_name: str,
) -> dict[str, Any]:
    # RQ2 fixes the scheduler config: p = p_max CPU workers and CPU batch equal
    # to the GPU batch (same batch policy, methods-scheduling). Policy tuning is
    # RQ3's question, so RQ2 runs no calibration sweep.
    cpu_workers = calibration_mod.max_cpu_workers(config.max_cpu_worker_candidate)
    cpu_batch_size = config.gpu_batch_size

    vanilla_times: dict[int, float] = {}
    hybrid_times: dict[int, float] = {}
    vanilla_records_first = None
    hybrid_records_first = None
    worker_split_first = None
    repeats_log: list[dict[str, Any]] = []

    for repeat in range(config.repeats):
        order = (
            ("vanilla", "hybrid")
            if repeat % 2 == 0
            else ("hybrid", "vanilla")
        )
        vanilla = None
        hybrid = None
        for candidate in order:
            if candidate == "vanilla":
                vanilla = baseline_candidate.run(
                    task, config.grid,
                    batch_size=config.gpu_batch_size, device=device, seed=config.seed,
                    gpu_slowdown_factor=slowdown,
                )
            elif candidate == "hybrid":
                hybrid = hybrid_candidate.run(
                    task, config.grid,
                    gpu_batch_size=config.gpu_batch_size,
                    cpu_batch_size=cpu_batch_size,
                    cpu_workers=cpu_workers,
                    device=device, seed=config.seed,
                    gpu_slowdown_factor=slowdown,
                )
        if vanilla is not None:
            vanilla_times[repeat] = vanilla.total_grid_s
            if vanilla_records_first is None:
                vanilla_records_first = vanilla.records
        if hybrid is not None:
            hybrid_times[repeat] = hybrid.total_grid_s
            if hybrid_records_first is None:
                hybrid_records_first = hybrid.records
                worker_split_first = hybrid.worker_throughput_split
        repeats_log.append(
            {
                "repeat": repeat,
                "trial_order": list(order),
                "vanilla_total_s": vanilla.total_grid_s if vanilla else None,
                "hybrid_total_s": hybrid.total_grid_s if hybrid else None,
            }
        )

    surface_validation = None
    if hybrid_records_first is not None and vanilla_records_first is not None:
        surface_validation = validate_surface(
            hybrid_records_first, vanilla_records_first, config.surface_gate,
        )
    surface_valid = bool(surface_validation["valid"]) if surface_validation else False

    speedups = paired_speedups(vanilla_times, hybrid_times)
    base_status, mean_, low, high = speedup_claim_status(
        speedups, surface_valid=surface_valid,
    )
    claim_status = _b_claim_status(
        base_status=base_status,
        surface_validation=surface_validation,
        r_native=r_native,
        regime_name=regime_name,
        ci_low=low,
    )

    achieved_ratio = _achieved_ratio(worker_split_first, vanilla_times, hybrid_times)
    slowdown_distance = abs((achieved_ratio or 1.0) - 1.0)

    return {
        "slowdown_factor": slowdown,
        "fixed_cell": {
            "gpu_batch_size": config.gpu_batch_size,
            "cpu_batch_size": cpu_batch_size,
            "cpu_workers": cpu_workers,
        },
        "vanilla": {
            "per_repeat_total_s": vanilla_times,
        },
        "hybrid": {
            "per_repeat_total_s": hybrid_times,
            "worker_throughput_split": worker_split_first,
        },
        "achieved_ratio": achieved_ratio,
        "slowdown_distance_from_unity": slowdown_distance,
        "surface_validation": surface_validation,
        "speedups": speedups,
        "speedup_mean": mean_,
        "speedup_ci_low": low,
        "speedup_ci_high": high,
        "claim_status": claim_status,
        "repeats": repeats_log,
    }


def _achieved_ratio(
    worker_split: dict[str, Any] | None,
    vanilla_times: dict[int, float],
    hybrid_times: dict[int, float],
) -> float | None:
    """Realized cpu/gpu throughput ratio under the regime's slowdown.
    Use the first repeat's worker split: (cpu_points / cpu_max_wall_s) / (gpu_points / gpu_wall_s)."""
    if not worker_split:
        return None
    cpu_pts = float(worker_split.get("cpu_points", 0) or 0)
    gpu_pts = float(worker_split.get("gpu_points", 0) or 0)
    cpu_wall = float(worker_split.get("cpu_max_wall_s", 0) or 0)
    gpu_wall = float(worker_split.get("gpu_wall_s", 0) or 0)
    if cpu_pts <= 0 or gpu_pts <= 0 or cpu_wall <= 0 or gpu_wall <= 0:
        return None
    cpu_throughput = cpu_pts / cpu_wall
    gpu_throughput = gpu_pts / gpu_wall
    return cpu_throughput / gpu_throughput


def _b_claim_status(
    *,
    base_status: str,
    surface_validation: dict[str, Any] | None,
    r_native: float,
    regime_name: str,
    ci_low: float | None,
) -> str:
    if surface_validation is not None and not surface_validation.get("valid", False):
        return "invalid_surface"
    if r_native >= 1.0 and regime_name == "native":
        # predictor falsification: r ≥ 1 but hybrid CI fails to exceed 1.0
        if ci_low is None or ci_low <= 1.0:
            if base_status in ("regression", "inconclusive"):
                return "predictor_invalid"
    if base_status == "speedup":
        return "hybrid_wins"
    if base_status == "regression":
        return "hybrid_regresses"
    return "inconclusive"
