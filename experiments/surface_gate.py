from __future__ import annotations

import math
from typing import Sequence

from experiments.schemas import SurfaceGateConfig, SurfaceValidation


SurfaceRecord = tuple[int, int, float]


def validate_surface(
    candidate: Sequence[SurfaceRecord],
    baseline: Sequence[SurfaceRecord],
    config: SurfaceGateConfig = SurfaceGateConfig(),
) -> SurfaceValidation:
    candidate_sorted = _sort_surface(candidate)
    baseline_sorted = _sort_surface(baseline)
    if len(candidate_sorted) != len(baseline_sorted):
        raise AssertionError(
            "surface point count differs: "
            f"{len(candidate_sorted)} != {len(baseline_sorted)}"
        )

    mismatch_count = 0
    first_mismatches: list[dict[str, float | int]] = []
    max_abs_error = 0.0
    squared_error_sum = 0.0

    for index, (candidate_record, baseline_record) in enumerate(
        zip(candidate_sorted, baseline_sorted)
    ):
        candidate_coord = candidate_record[:2]
        baseline_coord = baseline_record[:2]
        if candidate_coord != baseline_coord:
            raise AssertionError(
                "surface coordinate mismatch at sorted index "
                f"{index}: {candidate_coord} != {baseline_coord}"
            )

        row, col, candidate_loss = candidate_record
        _, _, baseline_loss = baseline_record
        abs_error = _absolute_error(candidate_loss, baseline_loss)
        max_abs_error = max(max_abs_error, abs_error)
        squared_error_sum += abs_error * abs_error

        if _losses_match(
            candidate_loss,
            baseline_loss,
            rel_tol=config.rel_tol,
            abs_tol=config.abs_tol,
        ):
            continue

        mismatch_count += 1
        if len(first_mismatches) < 10:
            first_mismatches.append(
                {
                    "row": row,
                    "col": col,
                    "lhs": candidate_loss,
                    "rhs": baseline_loss,
                    "abs_error": abs_error,
                }
            )

    point_count = len(candidate_sorted)
    rmse = math.sqrt(squared_error_sum / point_count) if point_count else 0.0
    return {
        "point_count": point_count,
        "mismatch_count": mismatch_count,
        "valid": mismatch_count == 0,
        "rel_tol": config.rel_tol,
        "abs_tol": config.abs_tol,
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "first_mismatches": first_mismatches,
    }


def _sort_surface(records: Sequence[SurfaceRecord]) -> list[SurfaceRecord]:
    return sorted(records, key=lambda record: (record[0], record[1]))


def _losses_match(
    lhs: float,
    rhs: float,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    if math.isnan(lhs) or math.isnan(rhs):
        return math.isnan(lhs) and math.isnan(rhs)
    if math.isinf(lhs) or math.isinf(rhs):
        return math.isinf(lhs) and math.isinf(rhs) and (lhs > 0) == (rhs > 0)
    return abs(lhs - rhs) <= max(rel_tol * max(abs(lhs), abs(rhs)), abs_tol)


def _absolute_error(lhs: float, rhs: float) -> float:
    if _same_nonfinite(lhs, rhs):
        return 0.0
    if not math.isfinite(lhs) or not math.isfinite(rhs):
        return math.inf
    return abs(lhs - rhs)


def _same_nonfinite(lhs: float, rhs: float) -> bool:
    if math.isnan(lhs) or math.isnan(rhs):
        return math.isnan(lhs) and math.isnan(rhs)
    if math.isinf(lhs) or math.isinf(rhs):
        return math.isinf(lhs) and math.isinf(rhs) and (lhs > 0) == (rhs > 0)
    return False
