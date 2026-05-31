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
from experiments.models import build_model, load_checkpoint
from experiments.schemas import GridSpec, MLTaskSpec


@dataclass(frozen=True)
class _WorkerSpec:
    label: str
    device_type: str
    batch_size: int
    candidate: GpuCandidate


@dataclass
class _WorkerState:
    device: torch.device
    model: torch.nn.Module
    evaluator: Any
    base: torch.Tensor
    direction_a: torch.Tensor
    direction_b: torch.Tensor


@dataclass
class _PoolWorkerHandle:
    spec: _WorkerSpec
    control_queue: Any
    process: mp.Process


def _build_worker_state(
    task: MLTaskSpec,
    grid: GridSpec,
    seed: int,
    spec: _WorkerSpec,
) -> _WorkerState:
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
    return _WorkerState(
        device=device,
        model=model,
        evaluator=evaluator,
        base=base,
        direction_a=dir_a,
        direction_b=dir_b,
    )


def _refresh_worker_state(
    state: _WorkerState,
    checkpoint_path: str | None,
    seed: int,
) -> None:
    if checkpoint_path is not None:
        load_checkpoint(state.model, checkpoint_path)
    state.model.eval()
    base_cpu, dir_a_cpu, dir_b_cpu = build_direction_vectors(state.model, seed)
    state.base.copy_(base_cpu.to(state.device))
    state.direction_a.copy_(dir_a_cpu.to(state.device))
    state.direction_b.copy_(dir_b_cpu.to(state.device))
    vector_to_parameters(state.base, state.model.parameters())
    device_mod.synchronize(state.device)


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
    state = _build_worker_state(task, grid, seed, spec)
    buffer_size = _buffer_size_for(spec.candidate)
    state.evaluator.warmup()
    device_mod.synchronize(state.device)

    local_records: Surface = []
    points_processed = 0
    wall_start = perf_counter()

    while True:
        buffer, sentinel = _pull_buffer(tasks_queue, buffer_size)
        if buffer:
            chunk_start = perf_counter()
            chunk_records = state.evaluator.evaluate(buffer)
            device_mod.synchronize(state.device)
            chunk_elapsed = perf_counter() - chunk_start
            device_mod.apply_gpu_slowdown(state.device, gpu_slowdown_factor, chunk_elapsed)
            device_mod.synchronize(state.device)
            local_records.extend(chunk_records)
            points_processed += len(buffer)
        if sentinel:
            break

    device_mod.synchronize(state.device)
    wall_s = perf_counter() - wall_start
    results_queue.put(
        {
            "label": spec.label,
            "device": state.device.type,
            "candidate": spec.candidate.name,
            "points_processed": points_processed,
            "records": local_records,
            "wall_s": wall_s,
            "diagnostics": state.evaluator.diagnostics(),
        }
    )


def _pool_worker_main(
    task: MLTaskSpec,
    grid: GridSpec,
    seed: int,
    spec: _WorkerSpec,
    control_queue,
    tasks_queue,
    results_queue,
    gpu_slowdown_factor: float,
) -> None:
    state = _build_worker_state(task, grid, seed, spec)
    buffer_size = _buffer_size_for(spec.candidate)
    cold_start_s = state.evaluator.warmup()
    device_mod.synchronize(state.device)
    results_queue.put(
        {
            "event": "ready",
            "label": spec.label,
            "device": state.device.type,
            "candidate": spec.candidate.name,
            "compile_cold_start_s": float(cold_start_s or 0.0),
            "diagnostics": state.evaluator.diagnostics(),
        }
    )

    while True:
        command = control_queue.get()
        if command is None or command.get("kind") == "close":
            break
        if command.get("kind") == "prepare":
            refresh_start = perf_counter()
            _refresh_worker_state(state, command.get("checkpoint_path"), seed)
            results_queue.put(
                {
                    "event": "prepared",
                    "label": spec.label,
                    "refresh_s": perf_counter() - refresh_start,
                }
            )
            continue
        if command.get("kind") != "run":
            raise RuntimeError(f"unknown hybrid pool command: {command!r}")

        local_records: Surface = []
        points_processed = 0
        wall_start = perf_counter()

        while True:
            buffer, sentinel = _pull_buffer(tasks_queue, buffer_size)
            if buffer:
                chunk_start = perf_counter()
                chunk_records = state.evaluator.evaluate(buffer)
                device_mod.synchronize(state.device)
                chunk_elapsed = perf_counter() - chunk_start
                device_mod.apply_gpu_slowdown(
                    state.device, gpu_slowdown_factor, chunk_elapsed
                )
                device_mod.synchronize(state.device)
                local_records.extend(chunk_records)
                points_processed += len(buffer)
            if sentinel:
                break

        device_mod.synchronize(state.device)
        wall_s = perf_counter() - wall_start
        results_queue.put(
            {
                "event": "result",
                "label": spec.label,
                "device": state.device.type,
                "candidate": spec.candidate.name,
                "points_processed": points_processed,
                "records": local_records,
                "wall_s": wall_s,
                "diagnostics": state.evaluator.diagnostics(),
            }
        )


def _make_worker_specs(
    *,
    gpu_batch_size: int,
    cpu_batch_size: int,
    cpu_workers: int,
    device: torch.device,
    gpu_candidate: GpuCandidate,
) -> list[_WorkerSpec]:
    if device.type == "cpu" and cpu_workers < 1:
        raise RuntimeError("CPU-only hybrid execution requires at least one CPU worker")

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
    return worker_specs


class HybridPool:
    """Session-scoped warm worker pool for steady-state hybrid grids."""

    def __init__(
        self,
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
    ) -> None:
        self._task = task
        self._grid = grid
        self._seed = seed
        self._gpu_slowdown_factor = gpu_slowdown_factor
        self._worker_specs = _make_worker_specs(
            gpu_batch_size=gpu_batch_size,
            cpu_batch_size=cpu_batch_size,
            cpu_workers=cpu_workers,
            device=device,
            gpu_candidate=gpu_candidate,
        )
        self._ctx = mp.get_context("spawn")
        self._tasks_queue = None
        self._results_queue = None
        self._workers: list[_PoolWorkerHandle] = []
        self._started = False
        self.compile_cold_start_s = 0.0
        self.pool_startup_s = 0.0
        self._diagnostics = {
            "candidate": gpu_candidate.name,
            "device": device.type,
            "gpu_batch_size": int(gpu_batch_size),
            "cpu_batch_size": int(cpu_batch_size),
            "cpu_workers": int(cpu_workers),
        }

    def __enter__(self) -> "HybridPool":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def start(self) -> "HybridPool":
        if self._started:
            return self
        self._tasks_queue = self._ctx.Queue()
        self._results_queue = self._ctx.Queue()
        startup_start = perf_counter()
        for spec in self._worker_specs:
            control_queue = self._ctx.Queue()
            process = self._ctx.Process(
                target=_pool_worker_main,
                args=(
                    self._task,
                    self._grid,
                    self._seed,
                    spec,
                    control_queue,
                    self._tasks_queue,
                    self._results_queue,
                    self._gpu_slowdown_factor,
                ),
            )
            process.start()
            self._workers.append(
                _PoolWorkerHandle(
                    spec=spec, control_queue=control_queue, process=process
                )
            )
        ready_payloads = self._collect_event("ready", len(self._workers))
        startup_wall_s = perf_counter() - startup_start
        self.compile_cold_start_s = max(
            (
                float(payload.get("compile_cold_start_s") or 0.0)
                for payload in ready_payloads
                if payload.get("label") == "gpu_0"
            ),
            default=0.0,
        )
        self.pool_startup_s = max(0.0, startup_wall_s - self.compile_cold_start_s)
        self._diagnostics = {
            **self._diagnostics,
            "compile_cold_start_s": self.compile_cold_start_s,
            "pool_startup_s": self.pool_startup_s,
            "worker_startup": ready_payloads,
        }
        self._started = True
        return self

    def close(self) -> None:
        if not self._workers:
            return
        for handle in self._workers:
            if handle.process.is_alive():
                handle.control_queue.put({"kind": "close"})
        for handle in self._workers:
            handle.process.join()
        self._workers = []
        self._started = False

    def run_grid(self, checkpoint_path: str | None = None) -> CandidateRunOutput:
        if not self._started:
            raise RuntimeError("HybridPool.start() must be called before run_grid()")
        assert self._tasks_queue is not None

        checkpoint_path = checkpoint_path or self._task.checkpoint_path
        for handle in self._workers:
            handle.control_queue.put(
                {"kind": "prepare", "checkpoint_path": checkpoint_path}
            )
        prepared_payloads = self._collect_event("prepared", len(self._workers))

        points = build_grid_points(self._grid)
        for point in points:
            self._tasks_queue.put(point)
        for _ in self._workers:
            self._tasks_queue.put(None)
        for handle in self._workers:
            handle.control_queue.put({"kind": "run"})

        payloads = self._collect_event("result", len(self._workers))
        return _output_from_worker_payloads(
            payloads,
            point_count=len(points),
            total_wall_s=None,
            diagnostics={
                **self._diagnostics,
                "checkpoint_path": checkpoint_path,
                "worker_refresh": prepared_payloads,
            },
            compile_cold_start_s=self.compile_cold_start_s,
        )

    def _collect_event(self, event: str, count: int) -> list[dict[str, Any]]:
        assert self._results_queue is not None
        payloads: list[dict[str, Any]] = []
        while len(payloads) < count:
            payload = self._results_queue.get()
            if payload.get("event") != event:
                raise RuntimeError(
                    f"expected hybrid pool event {event!r}, got {payload!r}"
                )
            payloads.append(payload)
        return payloads


def _output_from_worker_payloads(
    payloads: list[dict[str, Any]],
    *,
    point_count: int,
    total_wall_s: float | None,
    diagnostics: dict[str, Any],
    compile_cold_start_s: float | None = None,
) -> CandidateRunOutput:
    cpu_payloads = [p for p in payloads if p["label"].startswith("cpu_")]
    gpu_payloads = [p for p in payloads if p["label"] == "gpu_0"]
    cpu_points = sum(int(p["points_processed"]) for p in cpu_payloads)
    cpu_max_wall_s = max((p["wall_s"] for p in cpu_payloads), default=0.0)
    gpu_points = int(gpu_payloads[0]["points_processed"]) if gpu_payloads else 0
    gpu_wall_s = float(gpu_payloads[0]["wall_s"]) if gpu_payloads else 0.0

    records: Surface = []
    for payload in payloads:
        records.extend(payload["records"])

    worker_walls = [cpu_max_wall_s, gpu_wall_s]
    if total_wall_s is not None:
        worker_walls.append(total_wall_s)
    total_grid_s = max(worker_walls)
    total_points = max(1, point_count)
    worker_split = {
        "cpu_fraction": cpu_points / total_points,
        "cpu_points": cpu_points,
        "gpu_points": gpu_points,
        "cpu_max_wall_s": cpu_max_wall_s,
        "gpu_wall_s": gpu_wall_s,
    }
    recompile_count = max(
        (
            int((payload.get("diagnostics") or {}).get("recompile_count", 0) or 0)
            for payload in gpu_payloads
        ),
        default=0,
    )

    return CandidateRunOutput(
        records=records,
        total_grid_s=total_grid_s,
        compile_cold_start_s=compile_cold_start_s,
        recompile_count=recompile_count if gpu_payloads else None,
        worker_throughput_split=worker_split,
        diagnostics=diagnostics,
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
    points = build_grid_points(grid)
    ctx = mp.get_context("spawn")
    tasks_queue = ctx.Queue()
    results_queue = ctx.Queue()

    worker_specs = _make_worker_specs(
        gpu_batch_size=gpu_batch_size,
        cpu_batch_size=cpu_batch_size,
        cpu_workers=cpu_workers,
        device=device,
        gpu_candidate=gpu_candidate,
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

    return _output_from_worker_payloads(
        payloads,
        point_count=len(points),
        total_wall_s=total_wall_s,
        diagnostics={
            "candidate": gpu_candidate.name,
            "device": device.type,
            "gpu_batch_size": int(gpu_batch_size),
            "cpu_batch_size": int(cpu_batch_size),
            "cpu_workers": int(cpu_workers),
        },
    )
