from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from experiments.stats import speedup_claim_status


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
        candidates.append(
            {
                "workload": aggregate.get("workload_name"),
                "candidate": aggregate.get("candidate"),
                "role": aggregate.get("role"),
                "verdict": status,
                "speedup_geomean": geomean,
                "ci_low": low,
                "ci_high": high,
                "repeats": aggregate.get("repeats"),
            }
        )
    composition = [
        {
            "workload": workload,
            "status": entry.get("composition_status"),
            "ratio_geomean": entry.get("composition_ratio_mean"),
            "ci_low": entry.get("composition_ratio_ci_low"),
            "ci_high": entry.get("composition_ratio_ci_high"),
        }
        for workload, entry in (
            exp1.get("composition", {}).get("per_workload", {})
        ).items()
    ]
    record = exp1.get("record", {}) or {}
    return {
        "rq3_config": record.get("rq3_config"),
        "rq3_config_by_workload": record.get("rq3_config_by_workload"),
        "candidates": candidates,
        "composition": composition,
    }


def _project_rq2(exp2: dict) -> dict[str, Any]:
    workloads = []
    for name, workload in (exp2.get("record", {}).get("workloads", {})).items():
        predictor = workload.get("regime_predictor", {}) or {}
        regimes = []
        for regime_name, regime in (workload.get("regimes", {})).items():
            split = (regime.get("hybrid", {}) or {}).get(
                "worker_throughput_split", {}
            ) or {}
            regimes.append(
                {
                    "regime": regime_name,
                    "achieved_r": regime.get("achieved_ratio"),
                    "verdict": regime.get("claim_status"),
                    "ci_low": regime.get("speedup_ci_low"),
                    "ci_high": regime.get("speedup_ci_high"),
                    "cpu_fraction": split.get("cpu_fraction"),
                }
            )
        workloads.append(
            {
                "workload": name,
                "r_native": predictor.get("r_native"),
                "regimes": regimes,
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
