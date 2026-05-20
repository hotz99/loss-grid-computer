from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Two-tailed 95% CI t-critical values keyed by degrees of freedom (df = n - 1).
_T_CRIT_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
}


def _ci_95(values: list[float]) -> tuple[float, float] | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    stdev = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    t_crit = _T_CRIT_95.get(n - 1)
    if t_crit is None:
        return None
    half = t_crit * stdev / math.sqrt(n)
    return mean - half, mean + half


def _section_shares(experiments: dict) -> tuple[list[dict] | None, str | None]:
    exp = experiments.get("experiment_a_profiling") or {}
    if exp.get("status") != "completed":
        return None, "experiment_a_profiling not completed"
    workloads = ((exp.get("result") or {}).get("workloads")) or {}
    rows = []
    for name, wdata in workloads.items():
        st = (wdata.get("section_timings")) or {}
        total = st.get("total_grid_s") or 1.0
        rows.append({
            "workload": name,
            "perturbation_share": round((st.get("perturbation_s") or 0.0) / total, 4),
            "binding_share": round((st.get("binding_s") or 0.0) / total, 4),
            "batch_eval_share": round((st.get("batch_eval_s") or 0.0) / total, 4),
            "total_grid_s": round(total, 6),
        })
    return (rows or None), (None if rows else "no workload data in profiling result")


def _candidate_table(
    experiments: dict,
) -> tuple[list[dict] | None, list[str] | None, str | None]:
    exp = experiments.get("experiment_a_candidates") or {}
    if exp.get("status") != "completed":
        return None, None, "experiment_a_candidates not completed"
    scenarios = ((exp.get("result") or {}).get("scenarios")) or []
    if not scenarios:
        return None, None, "no scenarios in experiment_a_candidates result"

    rows: list[dict] = []
    issues: list[str] = []

    for scenario in scenarios:
        workload_name = (scenario.get("scenario") or {}).get("workload_name", "unknown")
        section_timings = (scenario.get("section_timings")) or {}

        for cm in (scenario.get("paired_candidate_metrics")) or []:
            candidate = cm.get("candidate") or "unknown"
            speedups: list[float] = [
                float(s) for s in (cm.get("paired_speedups") or []) if s is not None
            ]
            n = len(speedups)

            ci_lo = cm.get("paired_speedup_ci_95_lo")
            ci_hi = cm.get("paired_speedup_ci_95_hi")
            if (ci_lo is None or ci_hi is None) and n >= 2:
                ci = _ci_95(speedups)
                if ci:
                    ci_lo, ci_hi = ci

            if n < 2:
                issues.append(f"{workload_name}/{candidate}: n={n} < 2 — insufficient for CI")

            candidate_st = (section_timings.get(candidate)) or {}
            rows.append({
                "workload": workload_name,
                "candidate": candidate,
                "n": n,
                "mean_speedup": cm.get("paired_speedup_mean"),
                "stdev": cm.get("paired_speedup_stdev"),
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
                # Significant = lower CI bound strictly above 1.0 (beneficial)
                "significant": bool(ci_lo is not None and ci_lo > 1.0),
                "all_validations_passed": cm.get("all_validations_passed"),
                "section_timings_s": {
                    k: candidate_st.get(k)
                    for k in ("perturbation_s", "binding_s", "batch_eval_s", "total_grid_s")
                },
            })

    return (rows or None), (issues or None), (None if rows else "candidate table empty")


def _hybrid_table(experiments: dict) -> tuple[list[dict] | None, str | None]:
    exp = experiments.get("experiment_b_hybrid_applicability") or {}
    if exp.get("status") != "completed":
        return None, "experiment_b not completed"
    record_workloads = ((exp.get("record") or {}).get("workloads")) or {}
    if not record_workloads:
        return None, "no workload records in experiment_b result"
    rows = []
    for name, wdata in record_workloads.items():
        pp = (wdata.get("parity_probe")) or {}
        rows.append({
            "workload": name,
            "cpu_throughput_pts_s": wdata.get("cpu_throughput_points_per_s"),
            "gpu_throughput_pts_s": wdata.get("gpu_throughput_points_per_s"),
            "unslowed_ratio": wdata.get("unslowed_inference_ratio"),
            "slowdown_applied": pp.get("slowdown_used"),
            "achieved_ratio": pp.get("achieved_ratio"),
            "vanilla_time_s": pp.get("vanilla_runtime_s"),
            "selected_time_s": pp.get("selected_runtime_s"),
            "hybrid_speedup": pp.get("speedup"),
            "hybrid_wins": pp.get("hybrid_wins"),
            "surface_valid": pp.get("surface_valid"),
            "max_abs_error": pp.get("max_abs_error"),
            "worker_throughput_split": pp.get("worker_throughput_split"),
            "selected_policy": pp.get("selected_policy"),
        })
    return rows, None


def _calibration_workload_summary(workload_record: dict) -> dict:
    speedup = workload_record.get("session_speedup_vs_vanilla")
    return {
        "session_speedup_vs_vanilla": speedup,
        "speedup_exceeds_one": bool(speedup > 1.0) if speedup is not None else None,
        "break_even_n": workload_record.get("break_even_n"),
        "break_even_unavailable_reason": workload_record.get("break_even_unavailable_reason"),
        "amortized_calibration_per_variant_s": workload_record.get(
            "amortized_calibration_per_variant_s"
        ),
        "vanilla_per_variant_s": workload_record.get("vanilla_per_variant_s"),
        "hybrid_per_variant_grid_s": workload_record.get("hybrid_per_variant_grid_s"),
        "hybrid_without_cache_penalty_vs_vanilla": workload_record.get(
            "hybrid_without_cache_penalty_vs_vanilla"
        ),
        "selected_mode": workload_record.get("selected_policy"),
        "calibration_s": workload_record.get("calibration_s"),
        "calibration_grid_resolution": workload_record.get("calibration_grid_resolution"),
        "session_grid_resolution": workload_record.get("session_grid_resolution"),
        "variant_count": workload_record.get("variant_count"),
        "mode_selections_by_variant": workload_record.get("mode_selections_by_variant"),
        "session_total_s_vanilla": workload_record.get("session_total_s_vanilla"),
        "session_total_s_with_cache": workload_record.get("session_total_s_with_cache"),
        "session_total_s_without_cache": workload_record.get("session_total_s_without_cache"),
    }


def _calibration_summary(experiments: dict) -> tuple[dict | None, str | None]:
    exp = experiments.get("experiment_c_calibration_cache") or {}
    status = exp.get("status")
    if status == "skipped":
        return None, exp.get("reason") or "skipped"
    if status != "completed":
        return None, f"status={status or 'missing'}"
    record = (exp.get("record")) or {}
    workload_records = record.get("workloads") or {}
    if not workload_records:
        return None, "no workload records in experiment_c result"
    return {
        name: _calibration_workload_summary(wr)
        for name, wr in workload_records.items()
        if (wr.get("status")) == "completed"
    }, None


def collect(experiments: dict) -> dict[str, Any]:
    section_shares, shares_err = _section_shares(experiments)
    candidate_table, a_issues, candidates_err = _candidate_table(experiments)
    hybrid_table, b_err = _hybrid_table(experiments)
    calib_summary, c_err = _calibration_summary(experiments)

    gaps: list[str] = []
    if shares_err:
        gaps.append(f"section_shares: {shares_err}")
    if candidates_err:
        gaps.append(f"candidates: {candidates_err}")
    if a_issues:
        gaps.extend(a_issues)
    if b_err:
        gaps.append(f"exp_b: {b_err}")
    if c_err:
        gaps.append(f"exp_c: {c_err}")

    paper_ready = len(gaps) == 0

    n_significant = sum(
        1 for row in (candidate_table or []) if row.get("significant")
    )

    return {
        "paper_ready": paper_ready,
        "gaps": gaps,
        "n_significant_candidates": n_significant,
        "exp_a": {
            "section_shares": section_shares,
            "candidate_table": candidate_table,
        },
        "exp_b": {
            "workload_table": hybrid_table,
        },
        "exp_c": {
            "summary": calib_summary,
            "error": c_err,
        },
    }


def run(
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
    experiments: dict[str, Any],
) -> dict[str, Any]:
    del config
    del output_dir
    del shared_state
    result = collect(experiments)
    return {
        "status": "completed",
        "child_stem": "statistical-analysis",
        "result": result,
        "record": {
            "status": "completed",
            "paper_ready": result["paper_ready"],
            "gaps": result["gaps"],
            "n_significant_candidates": result["n_significant_candidates"],
        },
    }
