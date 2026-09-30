from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

import torch

from experiments.grid import GridPoint
from experiments.schemas import MLTaskSpec


Surface = list[tuple[int, int, float]]
Role = Literal["baseline", "vmapped", "compiled", "compiled_vmapped"]


@dataclass(frozen=True)
class GpuCandidate:
    role: Role
    point_chunk_size: int | None = None

    @classmethod
    def baseline(cls) -> "GpuCandidate":
        return cls(role="baseline")

    @classmethod
    def vmapped(cls, point_chunk_size: int) -> "GpuCandidate":
        return cls(role="vmapped", point_chunk_size=int(point_chunk_size))

    @classmethod
    def compiled(cls) -> "GpuCandidate":
        return cls(role="compiled")

    @classmethod
    def compiled_vmapped(cls, point_chunk_size: int) -> "GpuCandidate":
        return cls(role="compiled_vmapped", point_chunk_size=int(point_chunk_size))

    @property
    def name(self) -> str:
        if self.point_chunk_size is None:
            return self.role
        return f"{self.role}_k{self.point_chunk_size}"


@dataclass(frozen=True)
class CandidateRunOutput:
    records: Surface
    total_grid_s: float
    section_timings: dict[str, float] | None = None
    peak_cpu_memory_bytes: int | None = None
    peak_cuda_memory_bytes: int | None = None
    compile_cold_start_s: float | None = None
    recompile_count: int | None = None
    worker_throughput_split: dict[str, float] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ChunkEvaluator(Protocol):
    def warmup(self) -> float | None:
        """Trigger any one-time compilation; returns cold-start wall-time (s) or None."""

    def evaluate(self, chunk: Sequence[GridPoint]) -> Surface: ...

    def diagnostics(self) -> dict[str, Any]: ...


def make_chunk_evaluator(
    candidate: GpuCandidate,
    *,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    task: MLTaskSpec,
    base_device: torch.Tensor,
    direction_a_device: torch.Tensor,
    direction_b_device: torch.Tensor,
) -> ChunkEvaluator:
    if candidate.role == "baseline":
        from experiments.candidates.baseline import BaselineEvaluator

        return BaselineEvaluator(
            model=model, data_loader=data_loader, device=device, task=task,
            base=base_device, direction_a=direction_a_device, direction_b=direction_b_device,
        )
    if candidate.role == "vmapped":
        from experiments.candidates.vmapped import VmappedEvaluator

        if candidate.point_chunk_size is None:
            raise ValueError("vmapped candidate requires point_chunk_size")
        return VmappedEvaluator(
            model=model, data_loader=data_loader, device=device, task=task,
            base=base_device, direction_a=direction_a_device, direction_b=direction_b_device,
            point_chunk_size=candidate.point_chunk_size,
        )
    if candidate.role == "compiled":
        from experiments.candidates.compiled import CompiledEvaluator

        return CompiledEvaluator(
            model=model, data_loader=data_loader, device=device, task=task,
            base=base_device, direction_a=direction_a_device, direction_b=direction_b_device,
        )
    if candidate.role == "compiled_vmapped":
        from experiments.candidates.compiled_vmapped import CompiledVmappedEvaluator

        if candidate.point_chunk_size is None:
            raise ValueError("compiled_vmapped candidate requires point_chunk_size")
        return CompiledVmappedEvaluator(
            model=model, data_loader=data_loader, device=device, task=task,
            base=base_device, direction_a=direction_a_device, direction_b=direction_b_device,
            point_chunk_size=candidate.point_chunk_size,
        )
    raise ValueError(f"unsupported GpuCandidate role: {candidate.role}")
