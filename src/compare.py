from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

from src.results import load_run_summary, load_surface


def _resolve_run_dir(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        if candidate.name != "loss_surface.json":
            raise ValueError(
                f"Expected a run directory or loss_surface.json, got: {path}"
            )
        return candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"Run path does not exist: {path}")
    return candidate
def compare_run_outputs(
    lhs_path: str,
    rhs_path: str,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Dict[str, Any]:
    lhs_dir = _resolve_run_dir(lhs_path)
    rhs_dir = _resolve_run_dir(rhs_path)

    lhs_surface = load_surface(lhs_dir)
    rhs_surface = load_surface(rhs_dir)
    lhs_summary = load_run_summary(lhs_dir)
    rhs_summary = load_run_summary(rhs_dir)

    return compare_surfaces(
        lhs_surface=lhs_surface,
        rhs_surface=rhs_surface,
        atol=atol,
        rtol=rtol,
        lhs_total_s=lhs_summary.measurement.total_s,
        rhs_total_s=rhs_summary.measurement.total_s,
        lhs_run_dir=str(lhs_dir),
        rhs_run_dir=str(rhs_dir),
        lhs_device=lhs_summary.device,
        rhs_device=rhs_summary.device,
    )


def compare_surfaces(
    lhs_surface: list[tuple[int, int, float]],
    rhs_surface: list[tuple[int, int, float]],
    atol: float = 1e-6,
    rtol: float = 1e-5,
    lhs_total_s: Optional[float] = None,
    rhs_total_s: Optional[float] = None,
    lhs_run_dir: Optional[str] = None,
    rhs_run_dir: Optional[str] = None,
    lhs_device: Any = None,
    rhs_device: Any = None,
) -> Dict[str, Any]:
    lhs_sorted = sorted(lhs_surface, key=lambda record: (record[0], record[1]))
    rhs_sorted = sorted(rhs_surface, key=lambda record: (record[0], record[1]))
    if len(lhs_sorted) != len(rhs_sorted):
        raise ValueError(f"Point-count mismatch: {len(lhs_sorted)} vs {len(rhs_sorted)}")

    allclose = True
    mismatch_count = 0
    finite_squared_error_sum = 0.0
    finite_error_count = 0
    max_abs_error = 0.0
    for lhs_record, rhs_record in zip(lhs_sorted, rhs_sorted):
        lhs_row, lhs_col, lhs_value = lhs_record
        rhs_row, rhs_col, rhs_value = rhs_record
        if lhs_row != rhs_row or lhs_col != rhs_col:
            raise ValueError(
                f"Point mismatch: {(lhs_row, lhs_col)} vs {(rhs_row, rhs_col)}"
            )
        if math.isnan(lhs_value) and math.isnan(rhs_value):
            continue
        if math.isfinite(lhs_value) and math.isfinite(rhs_value):
            abs_err = abs(lhs_value - rhs_value)
            finite_squared_error_sum += abs_err ** 2
            finite_error_count += 1
            if abs_err > max_abs_error:
                max_abs_error = abs_err
        if not math.isclose(lhs_value, rhs_value, rel_tol=rtol, abs_tol=atol):
            allclose = False
            mismatch_count += 1
    rmse = math.sqrt(finite_squared_error_sum / max(1, finite_error_count))

    runtime_delta_s = None
    speedup_vs_lhs = None
    if lhs_total_s is not None and rhs_total_s is not None:
        runtime_delta_s = rhs_total_s - lhs_total_s
        if rhs_total_s > 0:
            speedup_vs_lhs = lhs_total_s / rhs_total_s

    return {
        "lhs_run_dir": lhs_run_dir,
        "rhs_run_dir": rhs_run_dir,
        "lhs_device": lhs_device,
        "rhs_device": rhs_device,
        "num_points": len(lhs_sorted),
        "allclose": allclose,
        "mismatch_count": mismatch_count,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "atol": atol,
        "rtol": rtol,
        "lhs_total_s": lhs_total_s,
        "rhs_total_s": rhs_total_s,
        "runtime_delta_s": runtime_delta_s,
        "speedup_rhs_vs_lhs_baseline": speedup_vs_lhs,
    }
