from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict, dataclass
import time
from typing import Sequence, Tuple

import torch

from src.backends.base import (
    build_output_dir,
    evaluate_points_on_device,
    make_surface,
)
from src.config import ExperimentConfig
from src.grid import GridPoint, partition_points
from src.kernel import build_execution_context, compile_chunk_evaluator
from src.results import ExperimentResult, Measurement, RunRecord


@dataclass(frozen=True)
class HybridCpuSchedulerLog:
    workers: int
    chunk_size_per_worker: int
    points_processed: dict[str, int]
    total_points_processed: int
    cpu_max_wall_time_s: float


@dataclass(frozen=True)
class HybridGpuSchedulerLog:
    initial_ratio: float
    chunk_size: int
    initial_points: int
    points_processed: int
    wall_time_s: float


@dataclass(frozen=True)
class HybridSchedulerLog:
    cpu: HybridCpuSchedulerLog
    gpu: HybridGpuSchedulerLog


def _evaluate_chunk(
    context, points: Sequence[GridPoint]
) -> list[Tuple[int, int, float]]:
    base_vector_device = context.base_vector_cpu.to(context.device)
    direction_a_device = context.direction_a_cpu.to(context.device)
    direction_b_device = context.direction_b_cpu.to(context.device)
    records = evaluate_points_on_device(
        context=context,
        points=points,
        base_vector_device=base_vector_device,
        direction_a_device=direction_a_device,
        direction_b_device=direction_b_device,
    )
    return records


def _cpu_worker_loop(
    config: ExperimentConfig,
    points: Sequence[GridPoint],
    next_index,
    lock,
    result_queue,
    worker_id: int,
) -> None:
    torch.set_num_threads(config.decomposition.cpu_threads_per_worker)
    context = build_execution_context(
        config, device_override="cpu", capture_env_info=False
    )
    local_records = []
    claimed_points = 0
    wall_start = time.perf_counter()
    chunk_count = 0
    total_points = len(points)
    chunk_size = config.decomposition.cpu_chunk_size

    while True:
        with lock:
            start = next_index.value
            if start >= total_points:
                break
            end = min(total_points, start + chunk_size)
            next_index.value = end

        if start >= end:
            break
        chunk = points[start:end]
        local_records.extend(_evaluate_chunk(context, chunk))

        claimed_points += len(chunk)
        chunk_count += 1

    total_wall_s = time.perf_counter() - wall_start

    print(f"[cpu_worker_{worker_id}] wall={total_wall_s:.4f}s chunks={chunk_count}")

    result_queue.put(
        {
            "worker_type": "cpu",
            "worker_id": worker_id,
            "points_processed": claimed_points,
            "records": local_records,
            "wall_s": total_wall_s,
        }
    )


def run(config):
    context = build_execution_context(config)
    surface = make_surface(config.grid.resolution)
    output_dir = build_output_dir(config)
    cpu_workers = config.resources.cpu_workers
    device = context.device
    points = partition_points(
        context.points,
        config.grid,
        worker_index=0,
        worker_count=1,
    )

    cpu_chunk_size = config.decomposition.cpu_chunk_size
    fixed_gpu_chunk_size = config.decomposition.fixed_gpu_chunk_size
    gpu_initial_ratio = config.decomposition.gpu_initial_ratio
    if fixed_gpu_chunk_size is not None:
        gpu_chunk_size = min(
            len(points),
            config.decomposition.gpu_chunk_size_max,
            fixed_gpu_chunk_size,
        )
    else:
        gpu_chunk_size = min(
            len(points), max(1, int(round(len(points) * gpu_initial_ratio)))
        )

    print(
        "[hybrid] allocation "
        f"total_points={len(points)} "
        f"gpu_initial_ratio={gpu_initial_ratio:.3f} "
        f"fixed_gpu_chunk_size={fixed_gpu_chunk_size} "
        f"gpu_initial_points={gpu_chunk_size} "
        f"cpu_chunk_size={cpu_chunk_size}"
    )

    surface = make_surface(config.grid.resolution)
    cpu_points = 0
    cpu_worker_points = {}
    cpu_max_wall_time_s = 0.0

    gpu_points_subset = list(points[:gpu_chunk_size])
    remaining_points = list(points[gpu_chunk_size:])
    cpu_helpers_enabled = bool(remaining_points) and cpu_workers > 0
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    workers = []
    if (
        context.device.type != "cpu"
        and context.compiled_gpu_chunk_eval_enabled
        and context.compiled_chunk_evaluator is None
        and isinstance(context.data_loader, list)
        and len(context.data_loader) > 0
    ):
        compile_chunk_evaluator(context)
    total_start = time.perf_counter()

    if cpu_helpers_enabled:
        next_index = ctx.Value("i", 0)
        lock = ctx.Lock()
        for worker_id in range(cpu_workers):
            process = ctx.Process(
                target=_cpu_worker_loop,
                args=(
                    config,
                    remaining_points,
                    next_index,
                    lock,
                    result_queue,
                    worker_id,
                ),
            )
            process.start()
            workers.append(process)

    gpu_points = 0
    gpu_worker_wall_s = 0.0
    gpu_worker_preload_s = context.preload_s
    gpu_worker_compile_s = context.compile_s

    try:
        gpu_wall_start = time.perf_counter()
        gpu_records = _evaluate_chunk(context, gpu_points_subset)
        gpu_worker_wall_s = time.perf_counter() - gpu_wall_start
        gpu_points = len(gpu_points_subset)
        for row, col, value in gpu_records:
            surface[row, col] = value

        print(
            f"[gpu_worker] preload={gpu_worker_preload_s:.4f}s "
            f"compile={gpu_worker_compile_s:.4f}s "
            f"wall={gpu_worker_wall_s:.4f}s "
            f"eval={gpu_worker_wall_s:.4f}s "
            f"lock_wait=0.0000s "
            f"idle=0.0% "
            f"chunks={1 if gpu_points_subset else 0}"
        )

        expected_messages = cpu_workers if cpu_helpers_enabled else 0
        for _ in range(expected_messages):
            payload = result_queue.get()
            cpu_points += payload["points_processed"]
            cpu_worker_points[f"worker_{payload['worker_id']}"] = payload[
                "points_processed"
            ]
            cpu_max_wall_time_s = max(cpu_max_wall_time_s, payload["wall_s"])
            for row, col, value in payload["records"]:
                surface[row, col] = value
    except KeyboardInterrupt:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join()
        raise
    else:
        for process in workers:
            process.join()

    total_runtime = time.perf_counter() - total_start

    result = ExperimentResult(
        record=RunRecord(
            experiment_name=config.experiment_name,
            measurement=Measurement(
                total_s=float(total_runtime),
                num_points=int(surface.numel()),
            ),
            backend=config.backend,
            device={
                "gpu": str(device),
                "cpu": config.resources.cpu_workers,
            },
            config=config.to_dict(),
            comparison=None,
            output_dir=output_dir,
        ),
        runtime_log={"total_s": total_runtime},
        surface=surface,
    )
    scheduler_log = HybridSchedulerLog(
        cpu=HybridCpuSchedulerLog(
            workers=cpu_workers,
            chunk_size_per_worker=cpu_chunk_size,
            points_processed=cpu_worker_points,
            total_points_processed=cpu_points,
            cpu_max_wall_time_s=cpu_max_wall_time_s,
        ),
        gpu=HybridGpuSchedulerLog(
            initial_ratio=gpu_initial_ratio,
            chunk_size=gpu_chunk_size,
            initial_points=len(gpu_points_subset),
            points_processed=gpu_points,
            wall_time_s=gpu_worker_wall_s,
        ),
    )
    result.runtime_log["hybrid_scheduler"] = asdict(scheduler_log)
    result.runtime_log["hybrid_scheduler"]["gpu"]["preload_s"] = gpu_worker_preload_s
    result.runtime_log["hybrid_scheduler"]["gpu"]["compile_s"] = gpu_worker_compile_s
    return result
