from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


SurfaceRecord = tuple[int, int, float]


@dataclass(frozen=True)
class SurfaceMismatch:
    row: int
    col: int
    lhs: float
    rhs: float
    abs_error: float


@dataclass(frozen=True)
class SurfaceComparison:
    point_count: int
    mismatch_count: int
    first_mismatches: tuple[SurfaceMismatch, ...]
    max_abs_error: float
    rmse: float
    allclose: bool
    rel_tol: float
    abs_tol: float


def sort_surface(records: Sequence[SurfaceRecord]) -> list[SurfaceRecord]:
    return sorted(records, key=lambda record: (record[0], record[1]))


def compare_surfaces(
    lhs_records: Sequence[SurfaceRecord],
    rhs_records: Sequence[SurfaceRecord],
    *,
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-6,
    max_mismatches: int = 10,
) -> SurfaceComparison:
    lhs_sorted = sort_surface(lhs_records)
    rhs_sorted = sort_surface(rhs_records)

    if len(lhs_sorted) != len(rhs_sorted):
        raise AssertionError(
            f"surface point count differs: {len(lhs_sorted)} != {len(rhs_sorted)}"
        )

    mismatch_count = 0
    first_mismatches: list[SurfaceMismatch] = []
    max_abs_error = 0.0
    squared_error_sum = 0.0

    for index, (lhs, rhs) in enumerate(zip(lhs_sorted, rhs_sorted)):
        lhs_coord = lhs[:2]
        rhs_coord = rhs[:2]
        if lhs_coord != rhs_coord:
            raise AssertionError(
                "surface coordinate mismatch at sorted index "
                f"{index}: {lhs_coord} != {rhs_coord}"
            )

        row, col, lhs_loss = lhs
        _, _, rhs_loss = rhs
        abs_error = _absolute_error(lhs_loss, rhs_loss)
        max_abs_error = max(max_abs_error, abs_error)
        squared_error_sum += abs_error * abs_error

        if _losses_match(lhs_loss, rhs_loss, rel_tol=rel_tol, abs_tol=abs_tol):
            continue

        mismatch_count += 1
        if len(first_mismatches) < max_mismatches:
            first_mismatches.append(
                SurfaceMismatch(
                    row=row,
                    col=col,
                    lhs=lhs_loss,
                    rhs=rhs_loss,
                    abs_error=abs_error,
                )
            )

    point_count = len(lhs_sorted)
    rmse = math.sqrt(squared_error_sum / point_count) if point_count else 0.0

    return SurfaceComparison(
        point_count=point_count,
        mismatch_count=mismatch_count,
        first_mismatches=tuple(first_mismatches),
        max_abs_error=max_abs_error,
        rmse=rmse,
        allclose=mismatch_count == 0,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


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
    return math.isclose(lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol)


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
