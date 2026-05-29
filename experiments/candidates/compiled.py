from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, Sequence

import torch
from torch.func import functional_call
from torch.nn.utils import vector_to_parameters

from experiments import device as device_mod
from experiments import memory as memory_mod
from experiments.candidates import _dynamo
from experiments.candidates._layout import (
    is_oom_error,
    make_functional_state,
)
from experiments.candidates.base import CandidateRunOutput, GpuCandidate, Surface
from experiments.data import build_dataloader, build_dataset
from experiments.grid import (
    GridPoint,
    build_direction_vectors,
    build_grid_points,
    perturb,
)
from experiments.loss import functional_loss
from experiments.models import build_model
from experiments.schemas import GridSpec, MLTaskSpec


def _flat_to_param_dict(flat_vector: torch.Tensor, layout) -> dict:
    return {
        entry.name: flat_vector.narrow(0, entry.offset, entry.numel).view(entry.shape)
        for entry in layout.entries
    }


def _move_batch(batch, device: torch.device):
    inputs, targets = batch
    # non_blocking is only safe from pinned host memory (CUDA dataloaders).
    # On MPS, async copies from pageable memory race the kernel and yield garbage.
    non_blocking = device.type == "cuda"
    return (
        inputs.to(device, dtype=torch.float32, non_blocking=non_blocking),
        targets.to(device, non_blocking=non_blocking),
    )


class CompiledEvaluator:
    """torch.compile on the per-point functional_call forward."""

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
    ) -> None:
        model.eval()
        self._model = model
        self._data_loader = data_loader
        self._device = device
        self._task = task
        self._base = base
        self._direction_a = direction_a
        self._direction_b = direction_b
        _, self._named_buffers, self._layout = make_functional_state(model)
        self._cold_start_s: float | None = None
        self._compiled_loss = torch.compile(self._loss_for_params, fullgraph=True)

    def _loss_for_params(self, params, inputs, targets):
        logits_or_predictions = functional_call(
            self._model, (params, self._named_buffers), (inputs,)
        )
        return functional_loss(logits_or_predictions, targets, self._task)

    def warmup(self) -> float | None:
        _dynamo.reset_counters()
        device_mod.synchronize(self._device)
        start = perf_counter()
        first_batch = next(iter(self._data_loader))
        inputs, targets = _move_batch(first_batch, self._device)
        params = _flat_to_param_dict(self._base, self._layout)
        with torch.no_grad():
            _ = self._compiled_loss(params, inputs, targets)
        device_mod.synchronize(self._device)
        self._cold_start_s = perf_counter() - start
        return self._cold_start_s

    def evaluate(self, chunk: Sequence[GridPoint]) -> Surface:
        records: Surface = []
        for point in chunk:
            perturbed = perturb(
                self._base, self._direction_a, self._direction_b, point.alpha, point.beta
            )
            params = _flat_to_param_dict(perturbed, self._layout)
            total_loss = 0.0
            total_examples = 0
            with torch.no_grad():
                for batch in self._data_loader:
                    inputs, targets = _move_batch(batch, self._device)
                    loss = self._compiled_loss(params, inputs, targets)
                    batch_size = int(targets.shape[0])
                    total_loss += float(loss.detach().cpu()) * batch_size
                    total_examples += batch_size
            records.append((point.row, point.col, total_loss / max(1, total_examples)))
        return records

    def diagnostics(self) -> dict[str, Any]:
        return {
            "compile_cold_start_s": self._cold_start_s,
            "recompile_count": _dynamo.recompile_count(),
        }


def run(
    task: MLTaskSpec,
    grid: GridSpec,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    gpu_slowdown_factor: float = 1.0,
) -> CandidateRunOutput:
    candidate = GpuCandidate.compiled()
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
        evaluator = CompiledEvaluator(
            model=model, data_loader=data_loader, device=device, task=task,
            base=base, direction_a=dir_a, direction_b=dir_b,
        )
        cold_start_s = evaluator.warmup()
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
                "failure_kind": "oom",
            },
            error=f"{type(exc).__name__}: {exc}",
        )

    return CandidateRunOutput(
        records=records,
        total_grid_s=total_grid_s,
        peak_cpu_memory_bytes=memory_mod.peak_cpu_bytes(),
        peak_cuda_memory_bytes=memory_mod.peak_cuda_bytes(device),
        compile_cold_start_s=cold_start_s,
        recompile_count=_dynamo.recompile_count(),
        diagnostics={"candidate": candidate.name, "device": device.type},
    )
