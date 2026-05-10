from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils import vector_to_parameters

from src.functional_eval.memory import (
    SectionTimings,
    cuda_memory_snapshot,
    process_memory_snapshot,
    reset_cuda_peak_memory,
)
from src.backends.base import build_grid_points
from src.original_algo import _prepare_reference_model_and_data
from src.results import synchronize_device
from src.system_schema import SchedulerRequest, VanillaMode


Surface = list[tuple[int, int, float]]


@dataclass(frozen=True)
class BaselineResult:
    candidate: str
    records: Surface
    timings: SectionTimings
    peak_cpu_memory_bytes: int | None
    peak_cuda_memory_bytes: int | None
    metadata: dict[str, Any]


def run_baseline(
    request: SchedulerRequest,
    *,
    seed: int = 1337,
) -> BaselineResult:
    if not isinstance(request.mode, VanillaMode):
        raise ValueError("baseline functional-eval wrapper expects VanillaMode")

    device = _resolve_device(request.device)
    (
        model,
        data_loader,
        base_vector_cpu,
        direction_a_cpu,
        direction_b_cpu,
    ) = _prepare_reference_model_and_data(request, device, seed=seed)

    points = build_grid_points(request.grid)
    base_vector_device = base_vector_cpu.to(device)
    direction_a_device = direction_a_cpu.to(device)
    direction_b_device = direction_b_cpu.to(device)
    vector_to_parameters(base_vector_device, model.parameters())

    if device.type == "cuda":
        reset_cuda_peak_memory(device)
    synchronize_device(device)

    import time

    perturbation_s = 0.0
    binding_s = 0.0
    batch_eval_s = 0.0
    records: Surface = []
    loss_fn = torch.nn.CrossEntropyLoss()

    total_started = time.perf_counter()
    for point in points:
        _sync(device)
        section_started = time.perf_counter()
        perturbed_variant = (
            base_vector_device
            + (point.alpha * direction_a_device)
            + (point.beta * direction_b_device)
        )
        _sync(device)
        perturbation_s += time.perf_counter() - section_started

        section_started = time.perf_counter()
        vector_to_parameters(perturbed_variant, model.parameters())
        model.eval()
        _sync(device)
        binding_s += time.perf_counter() - section_started

        total_loss = 0.0
        total_examples = 0
        section_started = time.perf_counter()
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                batch_size = int(targets.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size
        _sync(device)
        batch_eval_s += time.perf_counter() - section_started

        records.append((point.row, point.col, total_loss / max(1, total_examples)))

    synchronize_device(device)
    elapsed = time.perf_counter() - total_started

    cpu_memory = process_memory_snapshot()
    cuda_memory = cuda_memory_snapshot(device)
    peak_cuda_memory = (
        cuda_memory.max_reserved_bytes
        if cuda_memory.available
        else None
    )

    return BaselineResult(
        candidate="baseline_original",
        records=records,
        timings=SectionTimings(
            perturbation_s=perturbation_s,
            binding_s=binding_s,
            batch_eval_s=batch_eval_s,
            total_grid_s=elapsed,
        ),
        peak_cpu_memory_bytes=cpu_memory.rss_bytes if cpu_memory.available else None,
        peak_cuda_memory_bytes=peak_cuda_memory,
        metadata={
            "wrapped": "src.original_algo reference point-loop semantics",
            "device": device.type,
            "section_timings_available": True,
            "cpu_memory_available": cpu_memory.available,
            "cpu_memory_reason": cpu_memory.reason,
            "cuda_memory": _cuda_snapshot_to_dict(cuda_memory),
        },
    )


def _sync(device: torch.device) -> None:
    synchronize_device(device)


def _resolve_device(device: str) -> torch.device:
    return torch.device(
        device
        if device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )


def _cuda_snapshot_to_dict(snapshot: Any) -> dict[str, Any]:
    return {
        "available": snapshot.available,
        "allocated_bytes": snapshot.allocated_bytes,
        "reserved_bytes": snapshot.reserved_bytes,
        "max_allocated_bytes": snapshot.max_allocated_bytes,
        "max_reserved_bytes": snapshot.max_reserved_bytes,
        "reason": snapshot.reason,
    }
