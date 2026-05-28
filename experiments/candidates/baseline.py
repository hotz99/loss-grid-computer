from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

import torch
from torch.nn.utils import vector_to_parameters

from experiments import device as device_mod
from experiments import memory as memory_mod
from experiments.candidates.base import (
    CandidateRunOutput,
    GpuCandidate,
    Surface,
)
from experiments.data import build_dataloader, build_dataset
from experiments.grid import (
    GridPoint,
    build_direction_vectors,
    build_grid_points,
    perturb,
)
from experiments.loss import compute_loss
from experiments.models import build_model
from experiments.schemas import GridSpec, MLTaskSpec


class BaselineEvaluator:
    """In-place mutation + sequential variant loop (canon reference)."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        data_loader,
        device: torch.device,
        task: MLTaskSpec,
        base: torch.Tensor,
        direction_a: torch.Tensor,
        direction_b: torch.Tensor,
        profile_sections: bool = False,
    ) -> None:
        self._model = model
        self._data_loader = data_loader
        self._device = device
        self._task = task
        self._base = base
        self._direction_a = direction_a
        self._direction_b = direction_b
        self._profile_sections = profile_sections
        self._timings = {"perturbation_s": 0.0, "binding_s": 0.0, "batch_eval_s": 0.0}

    def warmup(self) -> float | None:
        return None

    def evaluate(self, chunk: Sequence[GridPoint]) -> Surface:
        records: Surface = []
        for point in chunk:
            t0 = perf_counter()
            perturbed = perturb(
                self._base, self._direction_a, self._direction_b, point.alpha, point.beta
            )
            if self._profile_sections:
                device_mod.synchronize(self._device)
                self._timings["perturbation_s"] += perf_counter() - t0

            t0 = perf_counter()
            vector_to_parameters(perturbed, self._model.parameters())
            self._model.eval()
            if self._profile_sections:
                device_mod.synchronize(self._device)
                self._timings["binding_s"] += perf_counter() - t0

            total_loss = 0.0
            total_examples = 0
            t0 = perf_counter()
            with torch.no_grad():
                for batch in self._data_loader:
                    loss, batch_size = compute_loss(
                        self._model, batch, self._device, self._task
                    )
                    total_loss += float(loss.cpu()) * batch_size
                    total_examples += batch_size
            if self._profile_sections:
                device_mod.synchronize(self._device)
                self._timings["batch_eval_s"] += perf_counter() - t0
            records.append((point.row, point.col, total_loss / max(1, total_examples)))
        return records

    def diagnostics(self) -> dict[str, Any]:
        if not self._profile_sections:
            return {}
        return dict(self._timings)


def run(
    task: MLTaskSpec,
    grid: GridSpec,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    profile_sections: bool = False,
    gpu_slowdown_factor: float = 1.0,
) -> CandidateRunOutput:
    return _run_impl(
        task, grid,
        batch_size=batch_size, device=device, seed=seed,
        profile_sections=profile_sections,
        gpu_slowdown_factor=gpu_slowdown_factor,
        gpu_candidate=GpuCandidate.baseline(),
    )


def _run_impl(
    task: MLTaskSpec,
    grid: GridSpec,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    profile_sections: bool,
    gpu_slowdown_factor: float,
    gpu_candidate: GpuCandidate,
) -> CandidateRunOutput:
    device_mod.seed_all(device, seed)
    model = build_model(task).to(device)
    model.eval()
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
    evaluator = BaselineEvaluator(
        model=model, data_loader=data_loader, device=device, task=task,
        base=base, direction_a=dir_a, direction_b=dir_b,
        profile_sections=profile_sections,
    )

    memory_mod.reset_peak_counters(device)
    device_mod.synchronize(device)
    start = perf_counter()
    records = evaluator.evaluate(points)
    device_mod.synchronize(device)
    eval_elapsed = perf_counter() - start
    device_mod.apply_gpu_slowdown(device, gpu_slowdown_factor, eval_elapsed)
    device_mod.synchronize(device)
    total_grid_s = perf_counter() - start

    section_timings = evaluator.diagnostics() or None
    if section_timings is not None:
        section_timings = {**section_timings, "total_grid_s": total_grid_s}

    return CandidateRunOutput(
        records=records,
        total_grid_s=total_grid_s,
        section_timings=section_timings,
        peak_cpu_memory_bytes=memory_mod.peak_cpu_bytes(),
        peak_cuda_memory_bytes=memory_mod.peak_cuda_bytes(device),
        diagnostics={"candidate": gpu_candidate.name, "device": device.type},
    )
