from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from torch.nn.utils import vector_to_parameters

from experiments import device as device_mod
from experiments.candidates.base import (
    CandidateRunOutput,
    GpuCandidate,
    Surface,
    make_chunk_evaluator,
)
from experiments.data import build_dataloader, build_dataset
from experiments.grid import (
    GridPoint,
    build_direction_vectors,
    build_grid_points,
)
from experiments.models import build_model
from experiments.schemas import GridSpec, MLTaskSpec


@dataclass(frozen=True)
class _WorkerSpec:
    label: str
    device_type: str
    batch_size: int
    candidate: GpuCandidate


def _build_worker_state(
    task: MLTaskSpec,
    grid: GridSpec,
    seed: int,
    spec: _WorkerSpec,
):
    device = device_mod.resolve(spec.device_type)
    device_mod.seed_all(device, seed)
    model = build_model(task).to(device)
    model.eval()
    dataset = build_dataset(task, seed)
    data_loader = build_dataloader(
        dataset, spec.batch_size, pin_memory=(device.type == "cuda")
    )
    base_cpu, dir_a_cpu, dir_b_cpu = build_direction_vectors(model, seed)
    base = base_cpu.to(device)
    dir_a = dir_a_cpu.to(device)
    dir_b = dir_b_cpu.to(device)
    vector_to_parameters(base, model.parameters())
    device_mod.synchronize(device)
    evaluator = make_chunk_evaluator(
        spec.candidate,
        model=model, data_loader=data_loader, device=device, task=task,
        base_device=base, direction_a_device=dir_a, direction_b_device=dir_b,
    )
    return device, evaluator


def _buffer_size_for(candidate: GpuCandidate) -> int:
    if candidate.role in ("vmapped", "compiled_vmapped"):
        return int(candidate.point_chunk_size or 1)
    return 1


def _pull_buffer(queue, buffer_size: int) -> tuple[list[GridPoint], bool]:
    buffer: list[GridPoint] = []
    saw_sentinel = False
    while len(buffer) < buffer_size:
        item = queue.get()
        if item is None:
            saw_sentinel = True
            break
        buffer.append(item)
    return buffer, saw_sentinel


def _worker_main(
    task: MLTaskSpec,
    grid: GridSpec,
    seed: int,
    spec: _WorkerSpec,
    tasks_queue,
    results_queue,
    gpu_slowdown_factor: float,
) -> None:
    device, evaluator = _build_worker_state(task, grid, seed, spec)
    buffer_size = _buffer_size_for(spec.candidate)
    evaluator.warmup()
    device_mod.synchronize(device)

    local_records: Surface = []
    points_processed = 0
    wall_start = perf_counter()

    while True:
        buffer, sentinel = _pull_buffer(tasks_queue, buffer_size)
        if buffer:
            chunk_start = perf_counter()
            chunk_records = evaluator.evaluate(buffer)
            device_mod.synchronize(device)
            chunk_elapsed = perf_counter() - chunk_start
            device_mod.apply_gpu_slowdown(device, gpu_slowdown_factor, chunk_elapsed)
            device_mod.synchronize(device)
            local_records.extend(chunk_records)
            points_processed += len(buffer)
        if sentinel:
            break

    device_mod.synchronize(device)
    wall_s = perf_counter() - wall_start
    results_queue.put(
        {
            "label": spec.label,
            "device": device.type,
            "candidate": spec.candidate.name,
            "points_processed": points_processed,
            "records": local_records,
            "wall_s": wall_s,
        }
    )


def run(
    task: MLTaskSpec,
    grid: GridSpec,
    *,
    gpu_batch_size: int,
    cpu_batch_size: int,
    cpu_workers: int,
    device: torch.device,
    seed: int,
    gpu_slowdown_factor: float = 1.0,
    gpu_candidate: GpuCandidate = GpuCandidate.baseline(),
) -> CandidateRunOutput:
    if device.type == "cpu" and cpu_workers < 1:
        raise RuntimeError("CPU-only hybrid execution requires at least one CPU worker")

    points = build_grid_points(grid)
    ctx = mp.get_context("spawn")
    tasks_queue = ctx.Queue()
    results_queue = ctx.Queue()

    worker_specs: list[_WorkerSpec] = []
    for worker_index in range(int(cpu_workers)):
        worker_specs.append(
            _WorkerSpec(
                label=f"cpu_{worker_index}",
                device_type="cpu",
                batch_size=cpu_batch_size,
                candidate=GpuCandidate.baseline(),  # canon: CPU runs baseline regardless of A
            )
        )
    if device.type != "cpu":
        worker_specs.append(
            _WorkerSpec(
                label="gpu_0",
                device_type=device.type,
                batch_size=gpu_batch_size,
                candidate=gpu_candidate,
            )
        )

    for point in points:
        tasks_queue.put(point)
    for _ in worker_specs:
        tasks_queue.put(None)

    wall_start = perf_counter()
    processes = []
    for spec in worker_specs:
        process = ctx.Process(
            target=_worker_main,
            args=(task, grid, seed, spec, tasks_queue, results_queue, gpu_slowdown_factor),
        )
        process.start()
        processes.append(process)

    payloads = [results_queue.get() for _ in processes]
    for process in processes:
        process.join()
    total_wall_s = perf_counter() - wall_start

    cpu_payloads = [p for p in payloads if p["label"].startswith("cpu_")]
    gpu_payloads = [p for p in payloads if p["label"] == "gpu_0"]
    cpu_points = sum(int(p["points_processed"]) for p in cpu_payloads)
    cpu_max_wall_s = max((p["wall_s"] for p in cpu_payloads), default=0.0)
    gpu_points = int(gpu_payloads[0]["points_processed"]) if gpu_payloads else 0
    gpu_wall_s = float(gpu_payloads[0]["wall_s"]) if gpu_payloads else 0.0

    records: Surface = []
    for payload in payloads:
        records.extend(payload["records"])

    total_grid_s = max(cpu_max_wall_s, gpu_wall_s, total_wall_s)
    total_points = max(1, len(points))
    worker_split = {
        "cpu_fraction": cpu_points / total_points,
        "cpu_points": cpu_points,
        "gpu_points": gpu_points,
        "cpu_max_wall_s": cpu_max_wall_s,
        "gpu_wall_s": gpu_wall_s,
    }

    return CandidateRunOutput(
        records=records,
        total_grid_s=total_grid_s,
        worker_throughput_split=worker_split,
        diagnostics={
            "candidate": gpu_candidate.name,
            "device": device.type,
            "gpu_batch_size": int(gpu_batch_size),
            "cpu_batch_size": int(cpu_batch_size),
            "cpu_workers": int(cpu_workers),
        },
    )
