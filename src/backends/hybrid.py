from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from typing import Any, Sequence, Tuple

import torch
from torch.nn.utils import vector_to_parameters

from src.backends.base import (
    apply_gpu_slowdown,
    build_grid_points,
    build_output_dir,
    evaluate_points_on_device,
    GridPoint,
    prepare_model_and_data,
    throughput,
)
from src.config import HybridExecutionConfig
from src.results import (
    DeviceRecord,
    ExperimentResult,
    Measurement,
    RunRecord,
    synchronize_device,
)


def _build_worker_context(
    config: HybridExecutionConfig,
    device: torch.device,
):
    (
        model,
        data_loader,
        preload_s,
        base_vector_cpu,
        direction_a_cpu,
        direction_b_cpu,
    ) = prepare_model_and_data(
        config,
        device,
    )

    base_vector_device = base_vector_cpu.to(device)
    direction_a_device = direction_a_cpu.to(device)
    direction_b_device = direction_b_cpu.to(device)
    vector_to_parameters(base_vector_device, model.parameters())
    synchronize_device(device)
    return (
        model,
        data_loader,
        preload_s,
        base_vector_device,
        direction_a_device,
        direction_b_device,
    )


def _execute_worker_queue(
    config: HybridExecutionConfig,
    device: torch.device,
    tasks,
    worker_label: str,
    worker_context,
):
    (
        model,
        data_loader,
        preload_s,
        base_vector_device,
        direction_a_device,
        direction_b_device,
    ) = worker_context
    claimed_points = 0
    chunk_count = 0
    local_records: list[Tuple[int, int, float]] = []
    synchronize_device(device)
    wall_start = time.perf_counter()

    while True:
        chunk = tasks.get()
        if chunk is None:
            break

        chunk_start = time.perf_counter()
        chunk_records = evaluate_points_on_device(
            config=config,
            model=model,
            data_loader=data_loader,
            device=device,
            chunk=chunk,
            base_vector_device=base_vector_device,
            direction_a_device=direction_a_device,
            direction_b_device=direction_b_device,
        )
        synchronize_device(device)
        chunk_runtime = time.perf_counter() - chunk_start

        apply_gpu_slowdown(
            device=device,
            gpu_slowdown_factor=config.workload.runtime.gpu_slowdown_factor,
            elapsed_s=chunk_runtime,
        )
        local_records.extend(
            (row, col, loss, worker_label) for row, col, loss in chunk_records
        )

        synchronize_device(device)
        claimed_points += len(chunk)
        chunk_count += 1
        if chunk_count % 8 == 0:
            elapsed_s = time.perf_counter() - wall_start
            print(
                f"[{worker_label}] progress "
                f"device={device.type} "
                f"chunks={chunk_count} points={claimed_points} "
                f"elapsed={elapsed_s:.4f}s"
            )

    synchronize_device(device)
    total_wall_s = time.perf_counter() - wall_start
    print(
        f"[{worker_label}] device={device.type} "
        f"wall={total_wall_s:.4f}s chunks={chunk_count}"
    )
    return {
        "worker_label": worker_label,
        "device": device.type,
        "points_processed": claimed_points,
        "records": local_records,
        "wall_s": total_wall_s,
        "preload_s": preload_s,
    }


def _worker_main(
    config: HybridExecutionConfig,
    device: torch.device,
    tasks,
    result_queue,
    worker_label: str,
):
    verbose = bool(config.workload.runtime.verbose)
    if not verbose:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    worker_context = _build_worker_context(
        config,
        device,
    )
    result_queue.put(
        _execute_worker_queue(
            config,
            device,
            tasks,
            worker_label,
            worker_context,
        )
    )


def _chunk_points(points: Sequence[GridPoint], chunk_size: int):
    size = max(1, int(chunk_size))
    return [list(points[index : index + size]) for index in range(0, len(points), size)]


def _resolve_gpu_device(config: HybridExecutionConfig):
    gpu_device = torch.device(
        config.workload.runtime.device
        if config.workload.runtime.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    if gpu_device.type == "cpu":
        raise RuntimeError("hybrid backend requires a GPU device")
    return gpu_device


def _build_task_queue(ctx, chunks, num_workers: int):
    tasks = ctx.Queue()
    for chunk in chunks:
        tasks.put(chunk)
    for _ in range(num_workers):
        tasks.put(None)
    return tasks


def _spawn_workers(
    ctx,
    config: HybridExecutionConfig,
    gpu_device: torch.device,
    tasks,
    result_queue,
):
    workers = []
    cpu_device = torch.device("cpu")

    for worker_index in range(config.cpu_workers):
        process = ctx.Process(
            target=_worker_main,
            args=(
                config,
                cpu_device,
                tasks,
                result_queue,
                f"cpu_{worker_index}",
            ),
        )
        process.start()
        workers.append(process)

    process = ctx.Process(
        target=_worker_main,
        args=(
            config,
            gpu_device,
            tasks,
            result_queue,
            "gpu_0",
        ),
    )
    process.start()
    workers.append(process)
    return workers


def _collect_payloads(workers, result_queue):
    payloads = [result_queue.get() for _ in workers]
    for process in workers:
        process.join()
    return payloads


def _summarize_payloads(
    config: HybridExecutionConfig,
    payloads,
    total_chunks: int,
):
    records: list[tuple[int, int, float]] = []
    worker_records: list[tuple[int, int, float, str]] = []
    worker_log: dict[str, dict[str, Any]] = {}

    for payload in payloads:
        worker_records.extend(payload["records"])
        records.extend((row, col, loss) for row, col, loss, _ in payload["records"])
        worker_log[payload["worker_label"]] = {
            "device": payload["device"],
            "points_processed": payload["points_processed"],
            "wall_time_s": payload["wall_s"],
            "preload_s": payload["preload_s"],
            "throughput_points_per_s": throughput(
                payload["points_processed"],
                payload["wall_s"],
            ),
        }

    gpu_log = worker_log["gpu_0"]
    cpu_logs = {
        label: log for label, log in worker_log.items() if label.startswith("cpu_")
    }
    cpu_points = sum(log["points_processed"] for log in cpu_logs.values())
    cpu_wall_time = max((log["wall_time_s"] for log in cpu_logs.values()), default=0.0)
    cpu_throughput = throughput(cpu_points, cpu_wall_time)

    print(
        f"[gpu_worker] preload={gpu_log['preload_s']:.4f}s "
        f"wall={gpu_log['wall_time_s']:.4f}s "
        f"eval={gpu_log['wall_time_s']:.4f}s "
        f"points={gpu_log['points_processed']}"
    )

    measurement = Measurement(
        total_s=max(gpu_log["wall_time_s"], cpu_wall_time),
        num_points=len(records),
    )
    runtime_log = {
        "total_s": measurement.total_s,
        "hybrid_execution": {
            "grid_compute_only_s": measurement.total_s,
            "points_processed": measurement.num_points,
            "throughput_points_per_s": measurement.get_points_per_s,
            "gpu_points_processed": gpu_log["points_processed"],
            "gpu_throughput_points_per_s": gpu_log["throughput_points_per_s"],
            "cpu_points_processed": cpu_points,
            "cpu_throughput_points_per_s": cpu_throughput,
        },
        "hybrid_scheduler": {
            "cpu": {
                "workers": config.cpu_workers,
                "points_processed": {
                    label: log["points_processed"] for label, log in cpu_logs.items()
                },
                "total_points_processed": cpu_points,
                "cpu_max_wall_time_s": cpu_wall_time,
            },
            "gpu": {
                "points_processed": gpu_log["points_processed"],
                "wall_time_s": gpu_log["wall_time_s"],
                "preload_s": gpu_log["preload_s"],
            },
            "queue": {
                "chunk_size": 1,
                "total_chunks": total_chunks,
            },
            "grid_compute_only_s": measurement.total_s,
        },
        "hybrid_records": [
            {"row": row, "col": col, "loss": loss, "worker": worker}
            for row, col, loss, worker in worker_records
        ],
    }

    print(
        "[hybrid_metrics] "
        f"grid_time={measurement.total_s:.4f}s "
        f"throughput={measurement.get_points_per_s:.4f}pts/s "
        f"gpu_throughput={gpu_log['throughput_points_per_s']:.4f}pts/s "
        f"cpu_throughput={cpu_throughput:.4f}pts/s"
    )
    return measurement, runtime_log, records, gpu_log


def run(config: HybridExecutionConfig):
    gpu_device = _resolve_gpu_device(config)

    points = build_grid_points(config.workload.grid)
    chunks = _chunk_points(points, 1)
    print(
        "[hybrid] allocation "
        f"total_points={len(points)} "
        f"gpu_device={gpu_device.type} "
        f"cpu_workers={config.cpu_workers} "
        "chunk_size=1"
    )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    tasks = _build_task_queue(ctx, chunks, 1 + config.cpu_workers)
    workers = _spawn_workers(
        ctx,
        config,
        gpu_device,
        tasks,
        result_queue,
    )
    payloads = _collect_payloads(workers, result_queue)
    measurement, runtime_log, records, gpu_log = _summarize_payloads(
        config,
        payloads,
        len(chunks),
    )

    return ExperimentResult(
        record=RunRecord(
            experiment_name=config.workload.experiment_name,
            measurement=measurement,
            backend=config._tag,
            device=DeviceRecord(gpu_log["device"], config.cpu_workers),
            config=config.workload,
            comparison=None,
            output_dir=build_output_dir(config),
        ),
        runtime_log=runtime_log,
        records=records,
    )
