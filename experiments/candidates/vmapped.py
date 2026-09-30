from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, Sequence

import torch
from torch.func import functional_call
from torch.nn.utils import vector_to_parameters

from experiments import device as device_mod
from experiments import memory as memory_mod
from experiments.candidates._layout import (
    flat_chunk_to_batched_param_dict,
    is_oom_error,
    make_functional_state,
    materialize_flat_chunk,
)
from experiments.candidates.base import CandidateRunOutput, GpuCandidate, Surface
from experiments.data import build_dataloader, build_dataset
from experiments.grid import (
    GridPoint,
    build_direction_vectors,
    build_grid_points,
)
from experiments.loss import functional_loss
from experiments.models import build_model
from experiments.schemas import GridSpec, MLTaskSpec


def _resolve_vmap():
    vmap = getattr(torch, "vmap", None)
    return vmap if vmap is not None else torch.func.vmap


def _move_batch(batch, device: torch.device):
    inputs, targets = batch
    # non_blocking is only safe from pinned host memory (CUDA dataloaders).
    # On MPS, async copies from pageable memory race the kernel and yield garbage.
    non_blocking = device.type == "cuda"
    return (
        inputs.to(device, dtype=torch.float32, non_blocking=non_blocking),
        targets.to(device, non_blocking=non_blocking),
    )


class VmappedEvaluator:
    """torch.func.functional_call + vmap over a chunk of perturbed parameter vectors."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        data_loader: Iterable,
        device: torch.device,
        task: MLTaskSpec,
        base: torch.Tensor,
        direction_a: torch.Tensor,
        direction_b: torch.Tensor,
        point_chunk_size: int,
    ) -> None:
        if point_chunk_size < 1:
            raise ValueError(f"point_chunk_size must be >= 1, got {point_chunk_size}")
        model.eval()
        self._model = model
        self._data_loader = data_loader
        self._device = device
        self._task = task
        self._base = base
        self._direction_a = direction_a
        self._direction_b = direction_b
        self._point_chunk_size = int(point_chunk_size)
        _, self._named_buffers, self._layout = make_functional_state(model)
        if int(base.numel()) != self._layout.total_numel:
            raise ValueError(
                "base length does not match model parameter layout: "
                f"expected {self._layout.total_numel}, got {int(base.numel())}"
            )
        self._vmap = _resolve_vmap()

    def warmup(self) -> float | None:
        return None

    def evaluate(self, chunk: Sequence[GridPoint]) -> Surface:
        records: Surface = []
        for sub in _chunks(chunk, self._point_chunk_size):
            records.extend(self._evaluate_sub(sub))
        return records

    def _evaluate_sub(self, sub: Sequence[GridPoint]) -> Surface:
        flat_vectors = materialize_flat_chunk(
            sub, self._base, self._direction_a, self._direction_b, self._device
        )
        batched_parameters = flat_chunk_to_batched_param_dict(flat_vectors, self._layout)
        sub_size = len(sub)
        weighted_loss_sum = torch.zeros(sub_size, device=self._device, dtype=torch.float32)
        total_examples = 0

        with torch.no_grad():
            for batch in self._data_loader:
                inputs, targets = _move_batch(batch, self._device)
                losses = self._vmap(
                    lambda params: self._loss_for_params(params, inputs, targets),
                    randomness="error",
                )(batched_parameters)
                batch_size = int(targets.shape[0])
                weighted_loss_sum += losses.detach().to(torch.float32) * batch_size
                total_examples += batch_size

        averages = (weighted_loss_sum / max(1, total_examples)).detach().cpu().tolist()
        return [(point.row, point.col, float(avg)) for point, avg in zip(sub, averages)]

    def _loss_for_params(self, params, inputs, targets):
        logits_or_predictions = functional_call(
            self._model, (params, self._named_buffers), (inputs,)
        )
        return functional_loss(logits_or_predictions, targets, self._task)

    def diagnostics(self) -> dict[str, Any]:
        return {"point_chunk_size": self._point_chunk_size}


def run(
    task: MLTaskSpec,
    grid: GridSpec,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    point_chunk_size: int,
    gpu_slowdown_factor: float = 1.0,
) -> CandidateRunOutput:
    candidate = GpuCandidate.vmapped(point_chunk_size)
    device_mod.seed_all(device, seed)
    model = build_model(task).to(device)
    dataset = build_dataset(task, seed)
    data_loader = build_dataloader(
        dataset, batch_size, pin_memory=(device.type == "cuda")
    )
    base_cpu, dir_a_cpu, dir_b_cpu = build_direction_vectors(model, seed)
    base = base_cpu.to(device)
    dir_a = dir_a_cpu.to(device)
    dir_b = dir_b_cpu.to(device)
    vector_to_parameters(base, model.parameters())
    device_mod.synchronize(device)

    points = build_grid_points(grid)
    memory_mod.reset_peak_counters(device)

    try:
        evaluator = VmappedEvaluator(
            model=model, data_loader=data_loader, device=device, task=task,
            base=base, direction_a=dir_a, direction_b=dir_b,
            point_chunk_size=point_chunk_size,
        )
        device_mod.synchronize(device)
        start = perf_counter()
        records = evaluator.evaluate(points)
        device_mod.synchronize(device)
        eval_elapsed = perf_counter() - start
        device_mod.apply_gpu_slowdown(device, gpu_slowdown_factor, eval_elapsed)
        device_mod.synchronize(device)
        total_grid_s = perf_counter() - start
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return CandidateRunOutput(
            records=[],
            total_grid_s=0.0,
            peak_cpu_memory_bytes=memory_mod.peak_cpu_bytes(),
            peak_cuda_memory_bytes=memory_mod.peak_cuda_bytes(device),
            diagnostics={
                "candidate": candidate.name,
                "device": device.type,
                "point_chunk_size": int(point_chunk_size),
                "failure_kind": "oom",
            },
            error=f"{type(exc).__name__}: {exc}",
        )

    return CandidateRunOutput(
        records=records,
        total_grid_s=total_grid_s,
        peak_cpu_memory_bytes=memory_mod.peak_cpu_bytes(),
        peak_cuda_memory_bytes=memory_mod.peak_cuda_bytes(device),
        diagnostics={
            "candidate": candidate.name,
            "device": device.type,
            "point_chunk_size": int(point_chunk_size),
        },
    )


def _chunks(points: Sequence[GridPoint], size: int) -> Iterable[Sequence[GridPoint]]:
    for start in range(0, len(points), size):
        yield points[start : start + size]
