from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

from src.backends import run_backend
from src.system_schema import (
    HybridMode,
    RunMode,
    RunRequest,
    SchedulerRequest,
    VanillaMode,
)


# ------------------------------
#            HELPERS
# ------------------------------


def resolve_available_cpu_cores() -> int:
    return max(1, os.cpu_count() or 1)


def resolve_cpu_worker_candidates() -> tuple[int, ...]:
    cpu_core_count = resolve_available_cpu_cores()
    values: list[int] = []
    candidate = 1

    while candidate < cpu_core_count:
        values.append(candidate)
        candidate *= 2

    if not values or values[-1] != cpu_core_count:
        values.append(cpu_core_count)

    return tuple(values)


def resolve_cpu_batch_size_candidates(
    sample_count: int,
    gpu_batch_size: int,
) -> tuple[int, ...]:
    upper_bound = max(1, min(sample_count, gpu_batch_size))
    if upper_bound <= 4:
        return (upper_bound,)

    values: list[int] = []
    candidate = 4
    while candidate < upper_bound:
        values.append(candidate)
        candidate *= 2

    if not values or values[-1] != upper_bound:
        values.append(upper_bound)

    return tuple(values)


def build_calibration_cache_key_payload(
    request: RunRequest,
    device: Literal["auto", "mps", "cuda", "cpu"],
    gpu_slowdown_factor: float,
    calibration_retry: int,
    calibration_cpu_workers: tuple[int, ...],
    calibration_cpu_batch_sizes: tuple[int, ...],
) -> dict[str, Any]:
    assert isinstance(request.mode, HybridMode)
    return {
        "schema_version": "calibration-cache-v1",
        "task_name": request.task.name,
        "model": request.task.model,
        "task": request.task.task,
        "loss": request.task.loss,
        "dataset_name": request.task.dataset.name,
        "dataset_input_shape": list(request.task.dataset.input_shape),
        "dataset_sample_count": request.task.dataset.sample_count,
        "grid_resolution": request.grid.resolution,
        "grid_scale": request.grid.scale,
        "device": device,
        "gpu_batch_size": request.mode.gpu_batch_size,
        "gpu_slowdown_factor": gpu_slowdown_factor,
        "calibration_retry": calibration_retry,
        "cpu_core_capacity": resolve_available_cpu_cores(),
        "calibration_cpu_workers": list(calibration_cpu_workers),
        "calibration_cpu_batch_sizes": list(calibration_cpu_batch_sizes),
    }


def resolve_calibration_cache_path(key_payload: dict[str, Any]) -> Path:
    digest = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Path("outputs") / "calibration-cache" / "v1" / f"{digest}.json"


def _mode_from_cache_payload(payload: dict[str, Any]) -> RunMode:
    if payload["_tag"] == "vanilla":
        return VanillaMode(int(payload["gpu_batch_size"]))
    return HybridMode(
        int(payload["gpu_batch_size"]),
        int(payload["cpu_batch_size"]),
        int(payload["cpu_workers"]),
    )


def load_calibration_cache(
    cache_path: Path,
):
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return (
        _mode_from_cache_payload(payload["resolved_mode"]),
        float(payload["baseline_total_s"]),
    )


def _mode_to_cache_payload(mode: RunMode) -> dict[str, Any]:
    if isinstance(mode, VanillaMode):
        return {
            "_tag": "vanilla",
            "gpu_batch_size": mode.gpu_batch_size,
        }
    return {
        "_tag": "hybrid",
        "gpu_batch_size": mode.gpu_batch_size,
        "cpu_batch_size": mode.cpu_batch_size,
        "cpu_workers": mode.cpu_workers,
    }


def write_calibration_cache(
    cache_path: Path,
    key_payload: dict[str, Any],
    resolved_mode: RunMode,
    baseline_total_s: float,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "calibration-cache-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "key_payload": key_payload,
        "baseline_total_s": baseline_total_s,
        "resolved_mode": _mode_to_cache_payload(resolved_mode),
    }
    cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ------------------------------
#          CALIBRATION
# ------------------------------


def run_calibration(
    request: RunRequest,
    baseline_total_s: float,
    cpu_worker_values: tuple[int, ...],
    cpu_batch_sizes: tuple[int, ...],
    retry: int,
    seed: int = 1337,
    gpu_slowdown_factor: float = 1.0,
) -> VanillaMode | HybridMode:
    assert isinstance(request.mode, HybridMode)
    records: list[tuple[float, HybridMode]] = []
    min_total_across_worker_counts = baseline_total_s
    consecutive_non_improvements = 0

    for cpu_workers in cpu_worker_values:
        min_total_for_worker_count_across_batch_sizes: Optional[float] = None

        for cpu_batch_size in cpu_batch_sizes:
            result = run_backend(
                SchedulerRequest(
                    request.task,
                    request.grid,
                    HybridMode(
                        request.mode.gpu_batch_size,
                        cpu_batch_size,
                        cpu_workers,
                    ),
                ),
                seed,
                gpu_slowdown_factor,
            )
            candidate_mode = HybridMode(
                request.mode.gpu_batch_size,
                cpu_batch_size,
                cpu_workers,
            )
            record: tuple[float, HybridMode] = (
                result.record.measurement.total_s,
                candidate_mode,
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
        return VanillaMode(request.mode.gpu_batch_size)
    return records[0][1]
