from __future__ import annotations

from typing import Dict, Optional


def throughput(num_points: int, total_runtime_s: float) -> float:
    if total_runtime_s <= 0:
        return 0.0
    return float(num_points) / float(total_runtime_s)


def strong_scaling_efficiency(
    baseline_time_s: Optional[float],
    current_time_s: float,
    workers: int,
) -> Optional[float]:
    if baseline_time_s is None or current_time_s <= 0 or workers <= 0:
        return None
    return baseline_time_s / (workers * current_time_s)


def weak_scaling_efficiency(
    baseline_time_s: Optional[float],
    current_time_s: float,
) -> Optional[float]:
    if baseline_time_s is None or current_time_s <= 0:
        return None
    return baseline_time_s / current_time_s


def build_metric_record(
    num_points: int,
    total_runtime_s: float,
    baseline_time_s_strong: Optional[float],
    baseline_time_s_weak: Optional[float],
    workers: int,
) -> Dict[str, Optional[float]]:
    return {
        "throughput_points_per_s": throughput(num_points, total_runtime_s),
        "strong_scaling_efficiency": strong_scaling_efficiency(
            baseline_time_s=baseline_time_s_strong,
            current_time_s=total_runtime_s,
            workers=workers,
        ),
        "weak_scaling_efficiency": weak_scaling_efficiency(
            baseline_time_s=baseline_time_s_weak,
            current_time_s=total_runtime_s,
        ),
    }
