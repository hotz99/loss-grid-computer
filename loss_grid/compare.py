from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def _resolve_run_dir(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        if candidate.name != "loss_surface.pt":
            raise ValueError(f"Expected a run directory or loss_surface.pt, got: {path}")
        return candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"Run path does not exist: {path}")
    return candidate


def _load_surface(run_dir: Path) -> torch.Tensor:
    surface_path = run_dir / "loss_surface.pt"
    if not surface_path.exists():
        raise FileNotFoundError(f"Missing surface file: {surface_path}")
    return torch.load(surface_path, map_location="cpu")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def compare_run_outputs(lhs_path: str, rhs_path: str, atol: float = 1e-6, rtol: float = 1e-5) -> Dict[str, Any]:
    lhs_dir = _resolve_run_dir(lhs_path)
    rhs_dir = _resolve_run_dir(rhs_path)

    lhs_surface = _load_surface(lhs_dir)
    rhs_surface = _load_surface(rhs_dir)
    if lhs_surface.shape != rhs_surface.shape:
        raise ValueError(f"Surface shape mismatch: {tuple(lhs_surface.shape)} vs {tuple(rhs_surface.shape)}")

    lhs_summary = _load_json(lhs_dir / "summary.json")
    rhs_summary = _load_json(rhs_dir / "summary.json")

    lhs_nan_mask = torch.isnan(lhs_surface)
    rhs_nan_mask = torch.isnan(rhs_surface)
    nan_mismatch_count = int(torch.count_nonzero(lhs_nan_mask != rhs_nan_mask).item())

    valid_mask = ~(lhs_nan_mask | rhs_nan_mask)
    valid_points = int(torch.count_nonzero(valid_mask).item())
    if valid_points == 0:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
        rmse = 0.0
        allclose = nan_mismatch_count == 0
    else:
        lhs_valid = lhs_surface[valid_mask]
        rhs_valid = rhs_surface[valid_mask]
        abs_diff = torch.abs(lhs_valid - rhs_valid)
        squared_diff = torch.square(lhs_valid - rhs_valid)
        max_abs_diff = float(abs_diff.max().item())
        mean_abs_diff = float(abs_diff.mean().item())
        rmse = float(torch.sqrt(squared_diff.mean()).item())
        allclose = nan_mismatch_count == 0 and bool(
            torch.allclose(lhs_valid, rhs_valid, atol=atol, rtol=rtol)
        )

    lhs_total_s = _float_or_none(lhs_summary.get("total_s"))
    rhs_total_s = _float_or_none(rhs_summary.get("total_s"))
    runtime_delta_s = None
    speedup_vs_lhs = None
    if lhs_total_s is not None and rhs_total_s is not None:
        runtime_delta_s = rhs_total_s - lhs_total_s
        if rhs_total_s > 0:
            speedup_vs_lhs = lhs_total_s / rhs_total_s

    return {
        "lhs_run_dir": str(lhs_dir),
        "rhs_run_dir": str(rhs_dir),
        "lhs_device": lhs_summary.get("device"),
        "rhs_device": rhs_summary.get("device"),
        "shape": list(lhs_surface.shape),
        "valid_points": valid_points,
        "nan_mismatch_count": nan_mismatch_count,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rmse": rmse,
        "allclose": allclose,
        "atol": atol,
        "rtol": rtol,
        "lhs_total_s": lhs_total_s,
        "rhs_total_s": rhs_total_s,
        "runtime_delta_s": runtime_delta_s,
        "speedup_rhs_vs_lhs_baseline": speedup_vs_lhs,
    }
