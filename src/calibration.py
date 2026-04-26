from __future__ import annotations

from typing import Literal, TypedDict, Optional

from src.backends import run_backend
from src.config import ExperimentConfig, HybridExecutionConfig


class BaselineCalibrationPolicy(TypedDict):
    _tag: Literal["vanilla"]


class HybridCpuPolicy(TypedDict):
    workers: int
    batch_size: int


class HybridCalibrationPolicy(TypedDict):
    _tag: Literal["hybrid_calibrated"]
    cpu: HybridCpuPolicy


CalibrationPolicy = BaselineCalibrationPolicy | HybridCalibrationPolicy


def run_calibration(
    config: ExperimentConfig,
    baseline_total_s: float,
    cpu_worker_values: tuple[int, ...],
    cpu_batch_sizes: tuple[int, ...],
    retry: int,
) -> CalibrationPolicy:
    records: list[tuple[float, HybridCalibrationPolicy]] = []
    min_total_across_worker_counts = baseline_total_s
    consecutive_non_improvements = 0

    for cpu_workers in cpu_worker_values:
        min_total_for_worker_count_across_batch_sizes: Optional[float] = None

        for cpu_batch_size in cpu_batch_sizes:
            result = run_backend(
                HybridExecutionConfig(
                    workload=config,
                    cpu_workers=cpu_workers,
                    cpu_batch_size=cpu_batch_size,
                )
            )
            hybrid_policy: HybridCalibrationPolicy = {
                "_tag": "hybrid_calibrated",
                "cpu": {"workers": cpu_workers, "batch_size": cpu_batch_size},
            }
            record: tuple[float, HybridCalibrationPolicy] = (
                result.record.measurement.total_s,
                hybrid_policy,
            )
            records.append(record)

            if (
                min_total_for_worker_count_across_batch_sizes is None
                or record[0] < min_total_for_worker_count_across_batch_sizes
            ):
                min_total_for_worker_count_across_batch_sizes = record[0]

        if (
            min_total_for_worker_count_across_batch_sizes
            < min_total_across_worker_counts
        ):
            min_total_across_worker_counts = (
                min_total_for_worker_count_across_batch_sizes
            )
            consecutive_non_improvements = 0
            continue

        consecutive_non_improvements += 1
        if consecutive_non_improvements >= retry:
            break

    records.sort(key=lambda record: record[0])
    if not records or records[0][0] >= baseline_total_s:
        return {"_tag": "baseline"}
    return {
        "_tag": "hybrid_calibrated",
        "cpu": records[0][1]["cpu"],
    }
