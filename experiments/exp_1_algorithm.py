from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from experiments import device as device_mod
from experiments.candidates import GpuCandidate, run_standalone
from experiments.candidates.base import CandidateRunOutput
from experiments.schemas import (
    CandidateAggregate,
    CandidateRunResult,
    Experiment1Config,
    Experiment1Result,
    TrialSpec,
)
from experiments.stats import (
    break_even_points,
    cold_inclusive_speedups,
    geometric_mean,
    mean,
    paired_speedups,
    speedup_claim_status,
    t_interval_95,
)
from experiments.surface_gate import validate_surface
from experiments.workloads import WORKLOADS, task_for_workload, workload_metadata


_SCHEMA_VERSION = "experiment-1-algorithm-v1"


@dataclass(frozen=True)
class _Candidate:
    name: str
    role: str
    gpu_candidate: GpuCandidate
    control: dict[str, int]


def plan(config: Experiment1Config) -> tuple[TrialSpec, ...]:
    candidates_by_role: dict[str, list[_Candidate]] = {}
    for candidate in _candidates(config):
        candidates_by_role.setdefault(candidate.role, []).append(candidate)

    trials: list[TrialSpec] = []
    for workload_name in config.workload_names:
        for repeat in range(config.repeats):
            trial_order = _trial_order(repeat, config.include_compile_candidates)
            for role in trial_order:
                for candidate in candidates_by_role.get(role, ()):
                    trials.append(
                        TrialSpec(
                            experiment="A",
                            workload_name=workload_name,
                            candidate=candidate.name,
                            repeat=repeat,
                            trial_order=trial_order,
                            control={"role": candidate.role, **candidate.control},
                        )
                    )
    return tuple(trials)


def run(config: Experiment1Config) -> Experiment1Result:
    _progress("start", workload_count=len(config.workload_names), repeats=config.repeats)
    device = device_mod.resolve(config.device)
    trials = plan(config)
    candidates = _candidates(config)
    _progress(
        "planned",
        device=device.type,
        trial_count=len(trials),
        candidate_count=len(candidates),
    )
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    candidate_roles: dict[str, list[str]] = {}
    for candidate in candidates:
        candidate_roles.setdefault(candidate.role, []).append(candidate.name)

    raw_times: dict[tuple[str, str, int], float] = {}
    raw_records: dict[tuple[str, str, int], list[tuple[int, int, float]]] = {}
    raw_diagnostics: dict[tuple[str, str, int], dict[str, Any]] = {}
    runs: list[CandidateRunResult] = []

    current_workload: str | None = None
    for trial in trials:
        if trial.workload_name != current_workload:
            current_workload = trial.workload_name
            _progress("workload", workload=current_workload)
        if trial.workload_name not in WORKLOADS:
            runs.append(
                CandidateRunResult(
                    workload_name=trial.workload_name,
                    candidate=trial.candidate,
                    role=str(trial.control["role"]),
                    repeat=trial.repeat,
                    status="unknown_workload",
                    trial_order=trial.trial_order,
                    diagnostics={"control": trial.control},
                )
            )
            continue
        task = task_for_workload(trial.workload_name, config.sample_count)
        spec = candidate_by_name[trial.candidate]
        try:
            output = run_standalone(
                spec.gpu_candidate, task, config.grid,
                batch_size=config.batch_size, device=device, seed=config.seed,
            )
        except Exception as exc:  # pragma: no cover - defensive
            runs.append(
                CandidateRunResult(
                    workload_name=trial.workload_name,
                    candidate=trial.candidate,
                    role=spec.role,
                    repeat=trial.repeat,
                    status="error",
                    trial_order=trial.trial_order,
                    error=f"{type(exc).__name__}: {exc}",
                    diagnostics={"control": trial.control},
                )
            )
            continue

        status = "error" if output.error else "ok"
        if status == "ok":
            raw_times[(trial.workload_name, trial.candidate, trial.repeat)] = output.total_grid_s
            raw_records[(trial.workload_name, trial.candidate, trial.repeat)] = output.records
        raw_diagnostics[(trial.workload_name, trial.candidate, trial.repeat)] = _trial_diagnostics(output)
        runs.append(
            CandidateRunResult(
                workload_name=trial.workload_name,
                candidate=trial.candidate,
                role=spec.role,
                repeat=trial.repeat,
                status=status,
                trial_order=trial.trial_order,
                total_grid_s=output.total_grid_s if status == "ok" else None,
                records=tuple(output.records),
                diagnostics={"control": trial.control, **_trial_diagnostics(output)},
                error=output.error,
            )
        )

    _progress("aggregate", workload_count=len(config.workload_names))
    aggregates = _aggregate(config, candidates, raw_times, raw_records, raw_diagnostics)
    composition = _composition(config, aggregates, raw_times)
    a_config_by_workload = {
        workload: _rq3_config(aggregates, workload)
        for workload in config.workload_names
    }
    workloads_record = {
        name: {
            "status": "planned" if name in WORKLOADS else "unknown_workload",
            "workload": workload_metadata(name, config.sample_count),
            "rq3_config": a_config_by_workload.get(name, "baseline"),
            "candidates": [
                _candidate_record(aggregate)
                for aggregate in aggregates
                if aggregate.workload_name == name
            ],
        }
        for name in config.workload_names
    }
    # Cross-workload rq3_config (single workload's value if only one;
    # otherwise the most-common value; ties broken by ordering in config).
    headline_a_config = _headline_rq3_config(a_config_by_workload)
    _progress("complete", rq3_config=headline_a_config, aggregate_count=len(aggregates))
    record = {
        "status": "completed",
        "implementation_status": "completed",
        "device": device.type,
        "trial_count": len(trials),
        "aggregate_count": len(aggregates),
        "candidate_count": len(candidates),
        "candidate_roles": candidate_roles,
        "rq3_config": headline_a_config,
        "rq3_config_by_workload": a_config_by_workload,
        "workloads": workloads_record,
    }
    return Experiment1Result(
        status="completed",
        schema_version=_SCHEMA_VERSION,
        config=config,
        trials=trials,
        runs=tuple(runs),
        aggregates=tuple(aggregates),
        rq3_config=headline_a_config,
        composition=composition,
        record=record,
    )


def _progress(event: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {payload}" if payload else ""
    print(f"[exp_1] {event}{suffix}", flush=True)


# --------------------------------------------------------------------------
#   Candidate construction
# --------------------------------------------------------------------------

def _candidates(config: Experiment1Config) -> tuple[_Candidate, ...]:
    candidates = [_Candidate("baseline", "baseline", GpuCandidate.baseline(), {})]
    for chunk in config.point_chunk_sizes:
        candidates.append(
            _Candidate(
                f"vmapped_k{chunk}", "vmapped",
                GpuCandidate.vmapped(chunk),
                {"point_chunk_size": chunk},
            )
        )
    if config.include_compile_candidates:
        candidates.append(
            _Candidate("compiled", "compiled", GpuCandidate.compiled(), {})
        )
        for chunk in config.point_chunk_sizes:
            candidates.append(
                _Candidate(
                    f"compiled_vmapped_k{chunk}", "compiled_vmapped",
                    GpuCandidate.compiled_vmapped(chunk),
                    {"point_chunk_size": chunk},
                )
            )
    return tuple(candidates)


def _trial_order(repeat: int, include_compile: bool) -> tuple[str, ...]:
    roles = (
        ("baseline", "vmapped", "compiled", "compiled_vmapped")
        if include_compile
        else ("baseline", "vmapped")
    )
    offset = repeat % len(roles)
    return roles[offset:] + roles[:offset]


# --------------------------------------------------------------------------
#   Aggregation
# --------------------------------------------------------------------------

def _trial_diagnostics(output: CandidateRunOutput) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "total_grid_s": output.total_grid_s,
        "peak_cpu_memory_bytes": output.peak_cpu_memory_bytes,
        "peak_cuda_memory_bytes": output.peak_cuda_memory_bytes,
    }
    if output.section_timings is not None:
        payload["section_timings"] = dict(output.section_timings)
    if output.compile_cold_start_s is not None:
        payload["compile_cold_start_s"] = output.compile_cold_start_s
    if output.recompile_count is not None:
        payload["recompile_count"] = output.recompile_count
    payload.update(output.diagnostics)
    return payload


def _aggregate(
    config: Experiment1Config,
    candidates: tuple[_Candidate, ...],
    raw_times: dict[tuple[str, str, int], float],
    raw_records: dict[tuple[str, str, int], list[tuple[int, int, float]]],
    raw_diagnostics: dict[tuple[str, str, int], dict[str, Any]],
) -> list[CandidateAggregate]:
    grid_points = config.grid.resolution * config.grid.resolution
    aggregates: list[CandidateAggregate] = []
    for workload in config.workload_names:
        baseline_times = {
            repeat: raw_times[(workload, "baseline", repeat)]
            for repeat in range(config.repeats)
            if (workload, "baseline", repeat) in raw_times
        }
        baseline_reference = next(
            (raw_records[(workload, "baseline", repeat)] for repeat in range(config.repeats)
             if (workload, "baseline", repeat) in raw_records),
            None,
        )
        for candidate in candidates:
            if candidate.role == "baseline":
                continue
            candidate_times = {
                repeat: raw_times[(workload, candidate.name, repeat)]
                for repeat in range(config.repeats)
                if (workload, candidate.name, repeat) in raw_times
            }
            surface_valid = True
            surface_validations: list[dict[str, Any]] = []
            for repeat in range(config.repeats):
                key = (workload, candidate.name, repeat)
                if key not in raw_records or baseline_reference is None:
                    continue
                validation = validate_surface(
                    raw_records[key], baseline_reference, config.surface_gate
                )
                surface_validations.append(validation)
                if not validation["valid"]:
                    surface_valid = False
            speedups = paired_speedups(baseline_times, candidate_times)
            status, mean_, low, high = speedup_claim_status(
                speedups, surface_valid=surface_valid
            )
            diagnostics: dict[str, Any] = {
                **candidate.control,
                "surface_valid": surface_valid,
                "surface_validations": surface_validations,
                "speedups": speedups,
            }
            amortization = _compile_amortization(
                workload, candidate.name, config, raw_diagnostics,
                baseline_times, candidate_times, grid_points,
            )
            if amortization is not None:
                diagnostics["compile_amortization"] = amortization
            aggregates.append(
                CandidateAggregate(
                    workload_name=workload,
                    candidate=candidate.name,
                    role=candidate.role,
                    speedup_mean=mean_,
                    speedup_ci_low=low,
                    speedup_ci_high=high,
                    claim_status=status,
                    repeats=len(speedups),
                    diagnostics=diagnostics,
                )
            )
    return aggregates


def _compile_amortization(
    workload: str,
    candidate: str,
    config: Experiment1Config,
    raw_diagnostics: dict[tuple[str, str, int], dict[str, Any]],
    baseline_times: dict[int, float],
    candidate_times: dict[int, float],
    grid_points: int,
) -> dict[str, Any] | None:
    """Compile-cost view for candidates that compile (compiled / compiled_vmapped).

    Steady-state speedup lives on the aggregate already (warm grid). Here we add
    the one-time compile cost, the cold-inclusive speedup at this grid size, and
    the break-even grid size where compile + warm evaluation overtakes the
    baseline. recompile_count is carried through as the witness that the warm
    grid timing was measured with no compilation leaking into it."""
    compile_times: dict[int, float] = {}
    recompiles: list[int] = []
    for repeat in range(config.repeats):
        diag = raw_diagnostics.get((workload, candidate, repeat))
        if not diag:
            continue
        cold = diag.get("compile_cold_start_s")
        if cold is not None:
            compile_times[repeat] = float(cold)
        rc = diag.get("recompile_count")
        if rc is not None:
            recompiles.append(int(rc))
    if not compile_times:
        return None

    cold_inclusive = cold_inclusive_speedups(
        baseline_times, candidate_times, compile_times
    )
    cold_interval = t_interval_95(cold_inclusive)
    break_even = break_even_points(
        baseline_times, candidate_times, compile_times, grid_points
    )
    break_even_geomean = geometric_mean(break_even)
    return {
        "compile_cold_start_s": [compile_times[r] for r in sorted(compile_times)],
        "compile_cold_start_mean_s": mean(compile_times.values()),
        "recompile_counts": recompiles,
        "recompile_count_max": max(recompiles) if recompiles else None,
        "grid_points": grid_points,
        "cold_inclusive_speedups": cold_inclusive,
        "cold_inclusive_geomean": geometric_mean(cold_inclusive),
        "cold_inclusive_ci_low": cold_interval[0] if cold_interval else None,
        "cold_inclusive_ci_high": cold_interval[1] if cold_interval else None,
        "break_even_points": break_even,
        "break_even_geomean": break_even_geomean,
        "amortizes_within_grid": (
            break_even_geomean is not None and break_even_geomean <= grid_points
        ),
    }


# --------------------------------------------------------------------------
#   Headline K + composition + winner
# --------------------------------------------------------------------------

def _rq3_config(aggregates: list[CandidateAggregate], workload: str) -> str:
    """Resolve selection rule per (workload, platform):
    highest supported speedup CI low bound, smaller K on ties for
    vmap-bearing candidates, baseline fallback."""
    qualified = [
        aggregate for aggregate in aggregates
        if aggregate.workload_name == workload
        and aggregate.speedup_ci_low is not None
        and aggregate.speedup_ci_low > 1.0
    ]
    if not qualified:
        return "baseline"
    def _k_for_tie_break(aggregate: CandidateAggregate) -> int:
        chunk = aggregate.diagnostics.get("point_chunk_size") if aggregate.diagnostics else None
        return int(chunk) if chunk is not None else 0
    return max(
        qualified,
        key=lambda aggregate: (
            aggregate.speedup_ci_low or 0.0,
            -_k_for_tie_break(aggregate),
        ),
    ).candidate


def _headline_rq3_config(a_config_by_workload: dict[str, str]) -> str:
    counts: dict[str, int] = {}
    for name in a_config_by_workload.values():
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "baseline"
    return max(counts, key=lambda key: counts[key])


def _composition(
    config: Experiment1Config,
    aggregates: list[CandidateAggregate],
    raw_times: dict[tuple[str, str, int], float],
) -> dict[str, Any]:
    """q_compose per (workload, K): speedup(compiled_vmapped_kK) / speedup(vmapped_kK),
    measured as the paired per-repeat ratio of grid times vmap_t / cv_t and reduced by
    geometric mean. It isolates the marginal benefit of torch.compile applied on top of
    vmap at a matched chunk size K, independent of whether compile alone is a standalone
    speedup. The verdict follows the standard CI taxonomy: improvement when the ratio CI
    low bound exceeds 1, regression when its high bound is below 1, otherwise unresolved
    (compile is neutral on top of vmap). The headline mirrors the K of the workload's
    rq3_config so it reflects the configuration RQ3 inherits."""
    per_workload: dict[str, dict[str, Any]] = {}
    for workload in config.workload_names:
        by_k: dict[str, Any] = {}
        for chunk in config.point_chunk_sizes:
            vmap_cand = f"vmapped_k{chunk}"
            cv_cand = f"compiled_vmapped_k{chunk}"
            ratios: list[float] = []
            for repeat in range(config.repeats):
                vmap_t = raw_times.get((workload, vmap_cand, repeat))
                cv_t = raw_times.get((workload, cv_cand, repeat))
                if not (vmap_t and cv_t):
                    continue
                ratios.append(vmap_t / cv_t)
            if not ratios:
                continue
            interval = t_interval_95(ratios)
            low = interval[0] if interval else None
            high = interval[1] if interval else None
            by_k[str(chunk)] = {
                "status": _composition_status(low, high),
                "ratio_geomean": geometric_mean(ratios),
                "ci_low": low,
                "ci_high": high,
                "vmapped_candidate": vmap_cand,
                "compiled_vmapped_candidate": cv_cand,
            }
        headline_k = _composition_headline_k(aggregates, workload, by_k)
        headline = by_k.get(headline_k, {}) if headline_k else {}
        per_workload[workload] = {
            "composition_status": headline.get("status", "unresolved"),
            "composition_ratio_mean": headline.get("ratio_geomean"),
            "composition_ratio_ci_low": headline.get("ci_low"),
            "composition_ratio_ci_high": headline.get("ci_high"),
            "headline_chunk_size": int(headline_k) if headline_k else None,
            "by_k": by_k,
        }
    return {"per_workload": per_workload}


def _chunk_from_candidate(candidate: str) -> int | None:
    marker = "_k"
    idx = candidate.rfind(marker)
    if idx == -1:
        return None
    tail = candidate[idx + len(marker):]
    return int(tail) if tail.isdigit() else None


def _composition_headline_k(
    aggregates: list[CandidateAggregate], workload: str, by_k: dict[str, Any],
) -> str | None:
    """Resolve the headline chunk size: the K of the workload's rq3_config when it is
    a vmap-bearing config measured here, otherwise the largest measured K."""
    if not by_k:
        return None
    chunk = _chunk_from_candidate(_rq3_config(aggregates, workload))
    if chunk is not None and str(chunk) in by_k:
        return str(chunk)
    return str(max(int(key) for key in by_k))


def _composition_status(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "unresolved"
    if low > 1.0:
        return "supported_improvement"
    if high < 1.0:
        return "supported_regression"
    return "unresolved"


def _candidate_record(aggregate: CandidateAggregate) -> dict[str, Any]:
    return {
        "candidate": aggregate.candidate,
        "role": aggregate.role,
        "claim_status": aggregate.claim_status,
        "speedup_mean": aggregate.speedup_mean,
        "speedup_ci_low": aggregate.speedup_ci_low,
        "speedup_ci_high": aggregate.speedup_ci_high,
        "repeats": aggregate.repeats,
        "diagnostics": aggregate.diagnostics,
    }
