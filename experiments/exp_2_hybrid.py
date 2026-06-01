from __future__ import annotations

from typing import Any

import torch

from experiments import device as device_mod
from experiments.candidates import baseline as baseline_candidate
from experiments.candidates import hybrid as hybrid_candidate
from experiments.cpu_resources import max_cpu_workers
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
# r_native is a per-point throughput ratio that only locates the expected
# crossing, so a small probe grid suffices.
_PROBE_GRID_RESOLUTION = 4


def _slowdown_ladder(config: Experiment2Config) -> tuple[int, ...]:
    """Base-2 geometric ladder {1, 2, 4, 8, ...} up to the configured ceiling,
    always swept in full from slow=1. The ladder start does not depend on
    r_native, so isolated-probe error cannot move the sweep past
    the true crossing."""
    rungs: list[int] = []
    rung = 1
    while rung <= config.slowdown_ceiling:
        rungs.append(rung)
        rung *= 2
    return tuple(rungs)


def plan(config: Experiment2Config) -> tuple[TrialSpec, ...]:
    trials: list[TrialSpec] = []
    for workload_name in config.workload_names:
        for slowdown in _slowdown_ladder(config):
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
                            control={"slowdown": slowdown},
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

    ladder: list[dict[str, Any]] = []
    for slowdown in _slowdown_ladder(config):
        _progress("rung", workload=workload_name, slowdown=slowdown)
        ladder.append(_run_rung(task, config, device, slowdown))

    threshold = _threshold_summary(ladder)

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
        "ladder": ladder,
        **threshold,
    }


def _threshold_summary(ladder: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce the swept ladder to the per-workload threshold summary.

    threshold_slowdown is the smallest rung whose paired CI clears 1.0
    (CI_low > 1.0). threshold_status distinguishes a native win, a win that
    needs slowdown, a ladder that never clears within the ceiling, and a
    monotonicity violation where a winning rung is followed by a regressing
    rung (CI_high < 1.0). threshold_bracket is the open-below interval
    (previous_rung, threshold_slowdown] that contains the true crossover.
    """
    crossing_index = None
    for index, rung in enumerate(ladder):
        ci_low = rung.get("speedup_ci_low")
        if isinstance(ci_low, (int, float)) and ci_low > 1.0:
            crossing_index = index
            break

    if crossing_index is None:
        return {
            "threshold_slowdown": None,
            "threshold_status": "above_explored_range",
            "threshold_bracket": None,
            "achieved_ratio_at_threshold": None,
        }

    # A win followed by a higher rung that regresses violates the monotonicity
    # the single-threshold report relies on.
    for rung in ladder[crossing_index + 1:]:
        ci_high = rung.get("speedup_ci_high")
        if isinstance(ci_high, (int, float)) and ci_high < 1.0:
            crossing = ladder[crossing_index]
            return {
                "threshold_slowdown": crossing["slowdown_factor"],
                "threshold_status": "non_monotone",
                "threshold_bracket": None,
                "achieved_ratio_at_threshold": crossing.get("achieved_ratio"),
            }

    crossing = ladder[crossing_index]
    threshold_slowdown = crossing["slowdown_factor"]
    if crossing_index == 0:
        return {
            "threshold_slowdown": threshold_slowdown,
            "threshold_status": "wins_at_native",
            "threshold_bracket": [None, threshold_slowdown],
            "achieved_ratio_at_threshold": crossing.get("achieved_ratio"),
        }
    previous_rung = ladder[crossing_index - 1]["slowdown_factor"]
    return {
        "threshold_slowdown": threshold_slowdown,
        "threshold_status": "crosses_within_range",
        "threshold_bracket": [previous_rung, threshold_slowdown],
        "achieved_ratio_at_threshold": crossing.get("achieved_ratio"),
    }


def _progress(event: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"[exp_2] {event}{suffix}", flush=True)


def _run_rung(
    task,
    config: Experiment2Config,
    device: torch.device,
    slowdown: float,
) -> dict[str, Any]:
    # RQ2 fixes the scheduler config: p = p_max CPU workers and CPU batch equal
    # to the GPU batch (same batch policy, methods-scheduling). Policy tuning is
    # RQ3's question, so RQ2 runs no composition-selection sweep.
    cpu_workers = max_cpu_workers(config.max_cpu_worker_candidate)
    cpu_batch_size = config.gpu_batch_size

    vanilla_times: dict[int, float] = {}
    hybrid_times: dict[int, float] = {}
    worker_split_first = None
    surface_pairs: list[dict[str, Any]] = []
    surface_validations: list[dict[str, Any]] = []
    repeats_log: list[dict[str, Any]] = []

    with hybrid_candidate.HybridPool(
        task,
        config.grid,
        gpu_batch_size=config.gpu_batch_size,
        cpu_batch_size=cpu_batch_size,
        cpu_workers=cpu_workers,
        device=device,
        seed=config.seed,
        gpu_slowdown_factor=slowdown,
    ) as pool:
        for repeat in range(config.repeats):
            order = (
                ("vanilla", "hybrid")
                if repeat % 2 == 0
                else ("hybrid", "vanilla")
            )
            vanilla = None
            vanilla_total_s = None
            hybrid = None
            for candidate in order:
                if candidate == "vanilla":
                    # The slowdown instrument appends a single post-eval idle
                    # delay with no concurrency on the vanilla path, so
                    # T_vanilla(slow) = slow * T_eval is the instrument's exact
                    # closed form. Measure T_eval at native r and scale, instead
                    # of sleeping (slow - 1) * T_eval of dead wall-clock per run.
                    # hybrid still runs the physical per-chunk delay because its
                    # scheduler overlaps the delay with CPU work.
                    vanilla = baseline_candidate.run(
                        task, config.grid,
                        batch_size=config.gpu_batch_size, device=device, seed=config.seed,
                        gpu_slowdown_factor=1.0,
                    )
                    vanilla_total_s = vanilla.total_grid_s * slowdown
                elif candidate == "hybrid":
                    hybrid = pool.run_grid(task.checkpoint_path)
            if vanilla is not None:
                vanilla_times[repeat] = vanilla_total_s
            if hybrid is not None:
                hybrid_times[repeat] = hybrid.total_grid_s
                if worker_split_first is None:
                    worker_split_first = hybrid.worker_throughput_split
            repeat_validation = None
            if hybrid is not None and vanilla is not None:
                surface_pairs.append(
                    {
                        "baseline": "vanilla",
                        "candidate": "hybrid",
                        "repeat": repeat,
                        "baseline_records": vanilla.records,
                        "candidate_records": hybrid.records,
                    }
                )
                repeat_validation = {
                    "repeat": repeat,
                    **validate_surface(
                        hybrid.records, vanilla.records, config.surface_gate,
                    ),
                }
                surface_validations.append(repeat_validation)
            repeats_log.append(
                {
                    "repeat": repeat,
                    "trial_order": list(order),
                    "vanilla_total_s": vanilla_total_s,
                    "hybrid_total_s": hybrid.total_grid_s if hybrid else None,
                    "surface_validation": repeat_validation,
                }
            )

    surface_validation = _surface_validation_summary(surface_validations)
    surface_valid = bool(surface_validation["valid"]) if surface_validation else False

    speedups = paired_speedups(vanilla_times, hybrid_times)
    base_status, mean_, low, high = speedup_claim_status(
        speedups, surface_valid=surface_valid,
    )
    claim_status = _b_claim_status(base_status)

    achieved_ratio = _achieved_ratio(worker_split_first, vanilla_times, hybrid_times)

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
        "surface_validation": surface_validation,
        "surface_validations": surface_validations,
        "surface_pairs": surface_pairs,
        "speedups": speedups,
        "speedup_mean": mean_,
        "speedup_ci_low": low,
        "speedup_ci_high": high,
        "claim_status": claim_status,
        "repeats": repeats_log,
    }


def _surface_validation_summary(
    validations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not validations:
        return None
    invalid = [item for item in validations if not item.get("valid")]
    return {
        "valid": not invalid,
        "repeat_count": len(validations),
        "valid_count": len(validations) - len(invalid),
        "invalid_count": len(invalid),
        "max_abs_error": max(
            float(item.get("max_abs_error") or 0.0)
            for item in validations
        ),
        "max_rmse": max(
            float(item.get("rmse") or 0.0)
            for item in validations
        ),
        "first_invalid": invalid[0] if invalid else None,
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


def _b_claim_status(base_status: str) -> str:
    """Per-rung verdict from the paired CI and the surface gate, in one pass.
    The predictor (r_native) plays no verdict role: it does not gate the sweep
    and cannot mark a rung invalid."""
    if base_status == "invalid_surface":
        return "invalid_surface"
    if base_status == "speedup":
        return "hybrid_wins"
    if base_status == "regression":
        return "hybrid_regresses"
    return "inconclusive"
