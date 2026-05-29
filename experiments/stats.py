from __future__ import annotations

import math
import statistics
from typing import Hashable, Iterable


_T_CRITICAL_95_TWO_SIDED = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.mean(items) if items else None


def stdev(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.stdev(items) if len(items) >= 2 else None


def geometric_mean(values: Iterable[float]) -> float | None:
    items = [float(v) for v in values if v > 0]
    if not items:
        return None
    return math.exp(statistics.mean([math.log(v) for v in items]))


def paired_speedups(
    baseline_times: dict[Hashable, float],
    candidate_times: dict[Hashable, float],
) -> list[float]:
    values: list[float] = []
    for repeat, baseline_s in baseline_times.items():
        candidate_s = candidate_times.get(repeat)
        if candidate_s is None or candidate_s <= 0:
            continue
        values.append(baseline_s / candidate_s)
    return values


def t_interval_95(values: Iterable[float]) -> tuple[float, float] | None:
    # Ratios are summarized on the log scale: the t-interval is built on
    # log(ratio) and exponentiated, so the center is the geometric mean and
    # the bounds respect the ratio's multiplicative geometry.
    items = [float(v) for v in values if v > 0]
    if len(items) < 2:
        return None
    logs = [math.log(v) for v in items]
    mean_log = statistics.mean(logs)
    stdev_log = statistics.stdev(logs)
    df = len(logs) - 1
    critical = _T_CRITICAL_95_TWO_SIDED.get(df, 1.96)
    half_width = critical * stdev_log / math.sqrt(len(logs))
    return math.exp(mean_log - half_width), math.exp(mean_log + half_width)


def speedup_claim_status(
    values: Iterable[float],
    *,
    surface_valid: bool,
) -> tuple[str, float | None, float | None, float | None]:
    items = list(values)
    sample_mean = geometric_mean(items)
    interval = t_interval_95(items)
    low = None if interval is None else interval[0]
    high = None if interval is None else interval[1]
    if not surface_valid:
        return "invalid_surface", sample_mean, low, high
    if interval is None:
        return "insufficient_data", sample_mean, low, high
    if low is not None and low > 1.0:
        return "speedup", sample_mean, low, high
    if high is not None and high < 1.0:
        return "regression", sample_mean, low, high
    return "inconclusive", sample_mean, low, high
