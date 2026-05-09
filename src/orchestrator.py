from __future__ import annotations

from typing import Literal

from src.backends import run_backend
from src.calibration import (
    build_calibration_cache_key_payload,
    load_calibration_cache,
    resolve_calibration_cache_path,
    run_calibration,
    resolve_cpu_batch_size_candidates,
    resolve_cpu_worker_candidates,
    write_calibration_cache,
)
from src.results import print_json
from src.schemas import (
    HybridMode,
    RunMode,
    RunRequest,
    SchedulerRequest,
    VanillaMode,
)


def run_orchestration(
    request: RunRequest,
    *,
    calibration_gpu_batch_size: int = 64,
    calibration_retry: int = 1,
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto",
    seed: int = 1337,
    gpu_slowdown_factor: float = 1.0,
):
    execution_seed = seed
    execution_slowdown = gpu_slowdown_factor
    resolved_mode: RunMode
    calibration_request = request

    if request.mode is None:
        calibration_request = RunRequest(
            request.task,
            request.grid,
            HybridMode(calibration_gpu_batch_size),
        )
        calibration_cpu_workers = resolve_cpu_worker_candidates()
        calibration_cpu_batch_sizes = resolve_cpu_batch_size_candidates(
            calibration_request.task.dataset.sample_count,
            calibration_gpu_batch_size,
        )

        cache_key_payload = build_calibration_cache_key_payload(
            calibration_request,
            device,
            gpu_slowdown_factor,
            calibration_retry,
            calibration_cpu_workers,
            calibration_cpu_batch_sizes,
        )
        calibration_cache_path = resolve_calibration_cache_path(cache_key_payload)
        cached_resolution = load_calibration_cache(calibration_cache_path)

        if cached_resolution is not None:
            resolved_mode, _ = cached_resolution
        else:
            baseline_result = run_backend(
                SchedulerRequest(
                    calibration_request.task,
                    calibration_request.grid,
                    VanillaMode(calibration_gpu_batch_size),
                    device,
                ),
                seed=execution_seed,
                gpu_slowdown_factor=execution_slowdown,
            )
            calibration_baseline_total_s = baseline_result.record.measurement.total_s
            resolved_mode = run_calibration(
                calibration_request,
                calibration_baseline_total_s,
                calibration_cpu_workers,
                calibration_cpu_batch_sizes,
                calibration_retry,
                execution_seed,
                execution_slowdown,
            )
            write_calibration_cache(
                calibration_cache_path,
                cache_key_payload,
                resolved_mode,
                calibration_baseline_total_s,
            )
    else:
        resolved_mode = request.mode

    print_json(
        run_backend(
            SchedulerRequest(
                request.task,
                request.grid,
                resolved_mode,
                device,
            ),
            seed=execution_seed,
            gpu_slowdown_factor=execution_slowdown,
        )
    )
