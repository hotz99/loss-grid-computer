from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from experiments.stats import geometric_mean, speedup_claim_status, t_interval_95


_SCHEMA_VERSION = "paper-projection-v2"


def project(exp1: dict, exp2: dict, exp3: dict) -> dict[str, Any]:
    """Reshape the three experiment JSON payloads into the paper-facing metrics.

    Inputs are the on-disk dict payloads (fresh-run or loaded from a results
    dir), never live dataclasses. RQ1 confidence intervals are recomputed from
    the stored per-repeat speedups so the projection always reflects the current
    stats.py; RQ2/RQ3 read the experiment-side fields.
    """
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "completed",
        "rq1": _project_rq1(exp1),
        "rq2": _project_rq2(exp2),
        "rq3": _project_rq3(exp3),
        "claims": _claims(exp1, exp2, exp3),
    }


def _project_rq1(exp1: dict) -> dict[str, Any]:
    candidates = []
    for aggregate in exp1.get("aggregates", []):
        diagnostics = aggregate.get("diagnostics", {}) or {}
        status, geomean, low, high = speedup_claim_status(
            diagnostics.get("speedups") or [],
            surface_valid=diagnostics.get("surface_valid", True),
        )
        entry = {
            "workload": aggregate.get("workload_name"),
            "candidate": aggregate.get("candidate"),
            "role": aggregate.get("role"),
            "verdict": status,
            "speedup_geomean": geomean,
            "ci_low": low,
            "ci_high": high,
            "repeats": aggregate.get("repeats"),
        }
        compile_cost = _compile_cost(diagnostics)
        if compile_cost is not None:
            entry["compile_cost"] = compile_cost
        candidates.append(entry)
    composition = _project_composition(exp1)
    record = exp1.get("record", {}) or {}
    return {
        "rq3_config": record.get("rq3_config"),
        "rq3_config_by_workload": record.get("rq3_config_by_workload"),
        "candidates": candidates,
        "composition": composition,
    }


def _chunk_from_candidate(candidate: str) -> int | None:
    marker = "_k"
    idx = candidate.rfind(marker)
    if idx == -1:
        return None
    tail = candidate[idx + len(marker):]
    return int(tail) if tail.isdigit() else None


def _composition_status(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "unresolved"
    if low > 1.0:
        return "supported_improvement"
    if high < 1.0:
        return "supported_regression"
    return "unresolved"


def _project_composition(exp1: dict) -> list[dict[str, Any]]:
    """Recompute q_compose from the stored per-repeat speedups so the projection
    tracks the current metric (matched-K compiled_vmapped over vmapped), mirroring
    how RQ1 confidence intervals are recomputed above. Per repeat the ratio is
    speedup(compiled_vmapped_kK) / speedup(vmapped_kK); both share the same per-repeat
    baseline so the baseline cancels and the ratio isolates the compile effect."""
    speedups: dict[tuple[str, str], list[float]] = {}
    workloads: list[str] = []
    for aggregate in exp1.get("aggregates", []):
        workload = aggregate.get("workload_name")
        candidate = aggregate.get("candidate")
        speedups[(workload, candidate)] = (
            (aggregate.get("diagnostics") or {}).get("speedups") or []
        )
        if workload not in workloads:
            workloads.append(workload)
    by_workload = (exp1.get("record", {}) or {}).get("rq3_config_by_workload") or {}

    composition: list[dict[str, Any]] = []
    for workload in workloads:
        chunks = sorted(
            chunk
            for chunk in {
                _chunk_from_candidate(candidate)
                for (current, candidate) in speedups
                if current == workload and candidate.startswith("vmapped_k")
            }
            & {
                _chunk_from_candidate(candidate)
                for (current, candidate) in speedups
                if current == workload
                and candidate.startswith("compiled_vmapped_k")
            }
            if chunk is not None
        )
        by_k: dict[str, Any] = {}
        for chunk in chunks:
            vmap = speedups.get((workload, f"vmapped_k{chunk}")) or []
            cv = speedups.get((workload, f"compiled_vmapped_k{chunk}")) or []
            pairs = min(len(vmap), len(cv))
            ratios = [cv[i] / vmap[i] for i in range(pairs) if vmap[i]]
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
            }
        headline_k = None
        if by_k:
            chunk = _chunk_from_candidate(by_workload.get(workload, "") or "")
            headline_k = (
                str(chunk)
                if chunk is not None and str(chunk) in by_k
                else str(max(int(key) for key in by_k))
            )
        headline = by_k.get(headline_k, {}) if headline_k else {}
        composition.append(
            {
                "workload": workload,
                "status": headline.get("status", "unresolved"),
                "ratio_geomean": headline.get("ratio_geomean"),
                "ci_low": headline.get("ci_low"),
                "ci_high": headline.get("ci_high"),
                "headline_chunk_size": int(headline_k) if headline_k else None,
                "by_k": by_k,
            }
        )
    return composition


def _compile_cost(diagnostics: dict) -> dict[str, Any] | None:
    """Paper-facing compile-cost view for candidates that compile. The
    cold-inclusive speedup CI and break-even grid size are recomputed from the
    stored per-repeat arrays so they track the current stats.py, mirroring how
    the steady-state speedup is recomputed above."""
    amort = (diagnostics or {}).get("compile_amortization")
    if not amort:
        return None
    cold_list = amort.get("cold_inclusive_speedups") or []
    cold_interval = t_interval_95(cold_list)
    break_even = geometric_mean(amort.get("break_even_points") or [])
    grid_points = amort.get("grid_points")
    return {
        "cold_start_mean_s": amort.get("compile_cold_start_mean_s"),
        "cold_inclusive_speedup_geomean": geometric_mean(cold_list),
        "cold_inclusive_ci_low": cold_interval[0] if cold_interval else None,
        "cold_inclusive_ci_high": cold_interval[1] if cold_interval else None,
        "break_even_grid_points": break_even,
        "grid_points": grid_points,
        "amortizes_within_grid": (
            break_even is not None
            and grid_points is not None
            and break_even <= grid_points
        ),
        "recompile_count_max": amort.get("recompile_count_max"),
    }


def _rq2_rung_verdict(low: float | None, high: float | None, surface_valid: bool) -> str:
    """Per-rung verdict from the recomputed CI and the surface gate, matching
    the rung taxonomy emitted by exp_2_hybrid (hybrid_wins / inconclusive /
    hybrid_regresses / invalid_surface)."""
    if not surface_valid:
        return "invalid_surface"
    if low is not None and low > 1.0:
        return "hybrid_wins"
    if high is not None and high < 1.0:
        return "hybrid_regresses"
    return "inconclusive"


def _rq2_threshold(ladder: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the threshold summary from the projected ladder CIs so the
    projection tracks the current stats.py, mirroring how RQ1 CIs are
    recomputed above. Smallest rung with ci_low > 1.0 is the threshold; a
    later regressing rung (ci_high < 1.0) flags non_monotone."""
    crossing = None
    for index, rung in enumerate(ladder):
        low = rung.get("ci_low")
        if isinstance(low, (int, float)) and low > 1.0:
            crossing = index
            break
    if crossing is None:
        return {
            "threshold_slowdown": None,
            "threshold_status": "above_explored_range",
            "threshold_bracket": None,
            "achieved_ratio_at_threshold": None,
        }
    for rung in ladder[crossing + 1:]:
        high = rung.get("ci_high")
        if isinstance(high, (int, float)) and high < 1.0:
            return {
                "threshold_slowdown": ladder[crossing]["slowdown"],
                "threshold_status": "non_monotone",
                "threshold_bracket": None,
                "achieved_ratio_at_threshold": ladder[crossing].get("achieved_r"),
            }
    slowdown = ladder[crossing]["slowdown"]
    if crossing == 0:
        return {
            "threshold_slowdown": slowdown,
            "threshold_status": "wins_at_native",
            "threshold_bracket": [None, slowdown],
            "achieved_ratio_at_threshold": ladder[crossing].get("achieved_r"),
        }
    return {
        "threshold_slowdown": slowdown,
        "threshold_status": "crosses_within_range",
        "threshold_bracket": [ladder[crossing - 1]["slowdown"], slowdown],
        "achieved_ratio_at_threshold": ladder[crossing].get("achieved_r"),
    }


def _project_rq2(exp2: dict) -> dict[str, Any]:
    workloads = []
    for name, workload in (exp2.get("record", {}).get("workloads", {})).items():
        predictor = workload.get("regime_predictor", {}) or {}
        ladder = []
        for rung in (workload.get("ladder", []) or []):
            split = (rung.get("hybrid", {}) or {}).get(
                "worker_throughput_split", {}
            ) or {}
            surface_valid = bool((rung.get("surface_validation") or {}).get("valid"))
            _, _, low, high = speedup_claim_status(
                rung.get("speedups") or [],
                surface_valid=surface_valid,
            )
            ladder.append(
                {
                    "slowdown": rung.get("slowdown_factor"),
                    "achieved_r": rung.get("achieved_ratio"),
                    "verdict": _rq2_rung_verdict(low, high, surface_valid),
                    "ci_low": low,
                    "ci_high": high,
                    "cpu_fraction": split.get("cpu_fraction"),
                }
            )
        workloads.append(
            {
                "workload": name,
                "r_native": predictor.get("r_native"),
                "ladder": ladder,
                **_rq2_threshold(ladder),
            }
        )
    return {
        "status": exp2.get("status"),
        "workloads": workloads,
    }


def _project_rq3(exp3: dict) -> dict[str, Any]:
    record = exp3.get("record", {}) or {}
    selection = record.get("rq2_selection", {}) or {}
    cell = record.get("selected_b_cell", {}) or {}
    return {
        "status": exp3.get("status"),
        "skip_reason": record.get("skip_reason"),
        "workload": selection.get("workload_name"),
        "operating_point": record.get("operating_point"),
        "slowdown_factor": record.get("slowdown_factor"),
        "rq3_config": record.get("rq3_config"),
        "selected_policy": cell.get("selected_policy"),
        "gpu_batch_size": cell.get("gpu_batch_size"),
        "cpu_workers": cell.get("cpu_workers"),
        "cpu_batch_size": cell.get("cpu_batch_size"),
        "calibration_s": record.get("calibration_s"),
        "compile_cold_start_s": record.get("compile_cold_start_s"),
        "a_config_compiles": record.get("a_config_compiles"),
        "one_time_setup_s": record.get("one_time_setup_s"),
        "T_v": record.get("T_v"),
        "T_p": record.get("T_p"),
        "cached_composed_session_total_s": record.get(
            "cached_composed_session_total_s"
        ),
        "session_speedup_vs_vanilla": record.get("session_speedup_vs_vanilla"),
        "break_even_n": record.get("break_even_n"),
        "amortization_label": record.get("amortization_label"),
    }


_NOT_READY = {"planned", "error", "disabled"}


def _ready(payload: dict) -> bool:
    return payload.get("status") not in _NOT_READY


def _claims(exp1: dict, exp2: dict, exp3: dict) -> dict[str, Any]:
    return {
        "rq1_ready": _ready(exp1),
        "rq2_ready": _ready(exp2),
        "rq3_ready": _ready(exp3),
    }


def _load_dir(results_dir: Path) -> dict[str, dict]:
    return {
        name: json.loads((results_dir / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("experiment-1", "experiment-2", "experiment-3")
    }


def main(argv: list[str]) -> Path:
    if len(argv) != 2:
        raise SystemExit("usage: python -m experiments.project <results_dir>")
    results_dir = Path(argv[1])
    data = _load_dir(results_dir)
    projection = project(
        data["experiment-1"], data["experiment-2"], data["experiment-3"]
    )
    out_path = results_dir / "projection.json"
    out_path.write_text(
        json.dumps(projection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(out_path)
    return out_path


if __name__ == "__main__":
    main(sys.argv)
