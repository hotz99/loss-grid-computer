from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import torch
from torch.func import functional_call

from src.backends.base import (
    GridPoint,
    Surface,
    build_grid_points,
    prepare_model_and_data,
    resolve_device,
)
from src.functional_eval.layout import (
    NamedTensorDict,
    flat_vector_to_param_dict,
    make_functional_state,
)
from src.functional_eval.memory import (
    CudaMemorySnapshot,
    ProcessMemorySnapshot,
    SectionTimings,
    cuda_memory_snapshot,
    process_memory_snapshot,
    reset_cuda_peak_memory,
)
from src.results import synchronize_device
from src.system_schema import SchedulerRequest, VanillaMode
from src.workloads import WORKLOADS, WorkloadDefinition


@dataclass(frozen=True)
class FunctionalSequentialResult:
    records: Surface
    timings: SectionTimings
    peak_cuda_memory: CudaMemorySnapshot
    process_memory: ProcessMemorySnapshot


class _FunctionalCallModule(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        params: NamedTensorDict,
        buffers: NamedTensorDict,
    ) -> None:
        super().__init__()
        self._module = module
        self._params = params
        self._buffers = buffers

    def forward(self, *args, **kwargs):
        return functional_call(
            self._module,
            (self._params, self._buffers),
            args,
            kwargs,
        )


def run_functional_sequential_surface(
    request: SchedulerRequest,
    *,
    seed: int = 1337,
) -> FunctionalSequentialResult:
    assert isinstance(request.mode, VanillaMode)
    torch_device = resolve_device(request.device)
    (
        model,
        data_loader,
        _preload_s,
        base_vector_cpu,
        direction_a_cpu,
        direction_b_cpu,
    ) = prepare_model_and_data(
        request,
        torch_device,
        seed,
    )

    points = build_grid_points(request.grid)
    base_vector_device = base_vector_cpu.to(torch_device)
    direction_a_device = direction_a_cpu.to(torch_device)
    direction_b_device = direction_b_cpu.to(torch_device)
    synchronize_device(torch_device)

    reset_cuda_peak_memory(torch_device)
    result = evaluate_points_functional(
        request=request,
        model=model,
        data_loader=data_loader,
        device=torch_device,
        chunk=points,
        base_vector_device=base_vector_device,
        direction_a_device=direction_a_device,
        direction_b_device=direction_b_device,
    )
    return FunctionalSequentialResult(
        records=result.records,
        timings=result.timings,
        peak_cuda_memory=cuda_memory_snapshot(torch_device),
        process_memory=process_memory_snapshot(),
    )


def evaluate_points_functional(
    request: SchedulerRequest,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    chunk: Sequence[GridPoint],
    base_vector_device: torch.Tensor,
    direction_a_device: torch.Tensor,
    direction_b_device: torch.Tensor,
) -> FunctionalSequentialResult:
    definition = WORKLOADS[request.task.name]
    return evaluate_points_functional_with_definition(
        definition=definition,
        model=model,
        data_loader=data_loader,
        device=device,
        chunk=chunk,
        base_vector_device=base_vector_device,
        direction_a_device=direction_a_device,
        direction_b_device=direction_b_device,
    )


def evaluate_points_functional_with_definition(
    *,
    definition: WorkloadDefinition,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    chunk: Sequence[GridPoint],
    base_vector_device: torch.Tensor,
    direction_a_device: torch.Tensor,
    direction_b_device: torch.Tensor,
) -> FunctionalSequentialResult:
    model.eval()
    _validate_vectors_share_layout(
        base_vector_device,
        direction_a_device,
        direction_b_device,
    )

    synchronize_device(device)
    total_started = perf_counter()
    _base_params, buffers, layout = make_functional_state(model)
    binding_s = _elapsed_since(total_started, device)

    records: Surface = []
    perturbation_s = 0.0
    batch_eval_s = 0.0

    with torch.inference_mode():
        for point in chunk:
            perturbation_started = perf_counter()
            perturbed_variant = (
                base_vector_device
                + (point.alpha * direction_a_device)
                + (point.beta * direction_b_device)
            )
            perturbation_s += _elapsed_since(perturbation_started, device)

            binding_started = perf_counter()
            point_params = flat_vector_to_param_dict(perturbed_variant, layout)
            functional_model = _FunctionalCallModule(model, point_params, buffers)
            functional_model.eval()
            binding_s += _elapsed_since(binding_started, device)

            total_loss = 0.0
            total_examples = 0
            batch_eval_started = perf_counter()
            for batch in data_loader:
                loss, batch_size = definition.compute_loss(
                    functional_model,
                    batch,
                    device,
                )
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size
            batch_eval_s += _elapsed_since(batch_eval_started, device)

            avg_loss = total_loss / max(1, total_examples)
            records.append((point.row, point.col, avg_loss))

    total_grid_s = _elapsed_since(total_started, device)
    return FunctionalSequentialResult(
        records=records,
        timings=SectionTimings(
            perturbation_s=float(perturbation_s),
            binding_s=float(binding_s),
            batch_eval_s=float(batch_eval_s),
            total_grid_s=float(total_grid_s),
        ),
        peak_cuda_memory=cuda_memory_snapshot(device),
        process_memory=process_memory_snapshot(),
    )


def _validate_vectors_share_layout(*vectors: torch.Tensor) -> None:
    if not vectors:
        return
    expected_shape = tuple(vectors[0].shape)
    expected_device = vectors[0].device
    for vector in vectors:
        if vector.ndim != 1:
            raise ValueError(
                f"expected flat 1D parameter vectors, got shape {tuple(vector.shape)}"
            )
        if tuple(vector.shape) != expected_shape:
            raise ValueError(
                "parameter vectors must have the same shape: "
                f"expected {expected_shape}, got {tuple(vector.shape)}"
            )
        if vector.device != expected_device:
            raise ValueError(
                "parameter vectors must be on the same device: "
                f"expected {expected_device}, got {vector.device}"
            )


def _elapsed_since(started_at_s: float, device: torch.device) -> float:
    synchronize_device(device)
    return perf_counter() - started_at_s
