import torch

from experiments.candidates.base import (
    CandidateRunOutput,
    ChunkEvaluator,
    GpuCandidate,
    make_chunk_evaluator,
)
from experiments.schemas import GridSpec, MLTaskSpec


__all__ = [
    "CandidateRunOutput",
    "ChunkEvaluator",
    "GpuCandidate",
    "make_chunk_evaluator",
    "run_standalone",
]


def run_standalone(
    candidate: GpuCandidate,
    task: MLTaskSpec,
    grid: GridSpec,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    gpu_slowdown_factor: float = 1.0,
) -> CandidateRunOutput:
    """Dispatch to the per-role standalone runner (A's experiment + C's gpu_only session)."""
    if candidate.role == "baseline":
        from experiments.candidates import baseline

        return baseline.run(
            task, grid, batch_size=batch_size, device=device, seed=seed,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
    if candidate.role == "vmapped":
        from experiments.candidates import vmapped

        if candidate.point_chunk_size is None:
            raise ValueError("vmapped candidate requires point_chunk_size")
        return vmapped.run(
            task, grid, batch_size=batch_size, device=device, seed=seed,
            point_chunk_size=candidate.point_chunk_size,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
    if candidate.role == "compiled":
        from experiments.candidates import compiled

        return compiled.run(
            task, grid, batch_size=batch_size, device=device, seed=seed,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
    if candidate.role == "compiled_vmapped":
        from experiments.candidates import compiled_vmapped

        if candidate.point_chunk_size is None:
            raise ValueError("compiled_vmapped candidate requires point_chunk_size")
        return compiled_vmapped.run(
            task, grid, batch_size=batch_size, device=device, seed=seed,
            point_chunk_size=candidate.point_chunk_size,
            gpu_slowdown_factor=gpu_slowdown_factor,
        )
    raise ValueError(f"unsupported GpuCandidate role: {candidate.role}")
