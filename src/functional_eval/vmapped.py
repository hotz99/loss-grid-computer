from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.func import functional_call

from src.backends.base import (
    GridPoint,
    Surface,
    build_grid_points,
    prepare_model_and_data,
)
from src.functional_eval.layout import (
    flat_chunk_to_batched_param_dict,
    make_functional_state,
)
from src.functional_eval.memory import (
    SectionTimings,
    cuda_memory_snapshot,
    process_memory_snapshot,
    reset_cuda_peak_memory,
)
from src.schemas import SchedulerRequest


@dataclass(frozen=True)
class VmappedEvalResult:
    candidate: str
    records: Surface
    timings: SectionTimings
    peak_cpu_memory_bytes: int | None
    peak_cuda_memory_bytes: int | None
    metadata: dict[str, Any]
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def run_vmapped_surface(
    request: SchedulerRequest,
    *,
    seed: int = 1337,
    point_chunk_size: int,
) -> VmappedEvalResult:
    """Build the requested workload and evaluate it with chunked vmap over points."""
    if point_chunk_size < 1:
        raise ValueError(f"point_chunk_size must be >= 1, got {point_chunk_size}")

    candidate = f"vmapped_functional_chunk_{point_chunk_size}"
    started = perf_counter()
    device = _resolve_device(request.device)
    try:
        (
            model,
            data_loader,
            _preload_s,
            base_vector_cpu,
            direction_a_cpu,
            direction_b_cpu,
        ) = prepare_model_and_data(request, device, seed=seed)

        return evaluate_vmapped_points(
            model=model,
            data_loader=data_loader,
            device=device,
            points=build_grid_points(request.grid),
            base_vector=base_vector_cpu.to(device),
            direction_a=direction_a_cpu.to(device),
            direction_b=direction_b_cpu.to(device),
            loss_name=request.task.loss,
            point_chunk_size=point_chunk_size,
        )
    except RuntimeError as exc:
        if not _is_oom_error(exc):
            raise
        _clear_cuda_cache(device)
        _synchronize_after_oom(device)
        return VmappedEvalResult(
            candidate=candidate,
            records=[],
            timings=SectionTimings(total_grid_s=perf_counter() - started),
            peak_cpu_memory_bytes=_peak_cpu_memory_bytes(),
            peak_cuda_memory_bytes=_peak_cuda_memory_bytes(device),
            metadata={
                "point_chunk_size": int(point_chunk_size),
                "device": device.type,
                "loss": request.task.loss,
                "failure_kind": "oom",
                "failure_stage": "candidate_setup",
            },
            error=f"{type(exc).__name__}: {exc}",
        )


def evaluate_vmapped_points(
    *,
    model: torch.nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    points: Sequence[GridPoint],
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    loss_name: str,
    point_chunk_size: int,
) -> VmappedEvalResult:
    """Evaluate a surface by vectorizing each data batch over perturbed model variants."""
    if point_chunk_size < 1:
        raise ValueError(f"point_chunk_size must be >= 1, got {point_chunk_size}")

    candidate = f"vmapped_functional_chunk_{point_chunk_size}"
    metadata: dict[str, Any] = {
        "point_chunk_size": int(point_chunk_size),
        "point_count": len(points),
        "device": device.type,
        "loss": loss_name,
    }
    reset_cuda_peak_memory(device if device.type == "cuda" else None)
    started = perf_counter()
    perturbation_s = 0.0
    binding_s = 0.0
    batch_eval_s = 0.0

    try:
        _synchronize(device)
        model.eval()
        _move_base_tensors_to_device(device, base_vector, direction_a, direction_b)
        named_parameters, named_buffers, layout = make_functional_state(model)
        if int(base_vector.numel()) != layout.total_numel:
            raise ValueError(
                "base_vector length does not match model parameter layout: "
                f"expected {layout.total_numel}, got {int(base_vector.numel())}"
            )

        records: Surface = []
        vmap = _resolve_vmap()

        for chunk in _chunks(points, point_chunk_size):
            perturbation_started = perf_counter()
            flat_vectors = _materialize_flat_chunk(
                chunk,
                base_vector,
                direction_a,
                direction_b,
                device,
            )
            _synchronize(device)
            perturbation_s += perf_counter() - perturbation_started

            binding_started = perf_counter()
            batched_parameters = flat_chunk_to_batched_param_dict(flat_vectors, layout)
            _synchronize(device)
            binding_s += perf_counter() - binding_started

            chunk_size = len(chunk)
            accumulator_dtype = (
                torch.float32 if device.type == "mps" else torch.float64
            )
            weighted_loss_sum = torch.zeros(
                chunk_size,
                device=device,
                dtype=accumulator_dtype,
            )
            total_examples = 0

            with torch.no_grad():
                for batch in data_loader:
                    batch_started = perf_counter()
                    inputs, targets = _move_batch(batch, device)
                    losses = vmap(
                        lambda params: _loss_for_params(
                            model,
                            named_buffers,
                            params,
                            inputs,
                            targets,
                            loss_name,
                        ),
                        randomness="error",
                    )(batched_parameters)
                    batch_size = int(targets.shape[0])
                    weighted_loss_sum += losses.detach().to(accumulator_dtype) * batch_size
                    total_examples += batch_size
                    _synchronize(device)
                    batch_eval_s += perf_counter() - batch_started

            averages = (
                weighted_loss_sum / max(1, total_examples)
            ).detach().cpu().tolist()
            for point, avg_loss in zip(chunk, averages):
                records.append((point.row, point.col, float(avg_loss)))

        _synchronize(device)
        total_grid_s = perf_counter() - started
        return VmappedEvalResult(
            candidate=candidate,
            records=records,
            timings=SectionTimings(
                perturbation_s=perturbation_s,
                binding_s=binding_s,
                batch_eval_s=batch_eval_s,
                total_grid_s=total_grid_s,
            ),
            peak_cpu_memory_bytes=_peak_cpu_memory_bytes(),
            peak_cuda_memory_bytes=_peak_cuda_memory_bytes(device),
            metadata={
                **metadata,
                "parameter_count": len(named_parameters),
                "buffer_count": len(named_buffers),
            },
        )
    except RuntimeError as exc:
        if not _is_oom_error(exc):
            raise
        _clear_cuda_cache(device)
        _synchronize_after_oom(device)
        total_grid_s = perf_counter() - started
        return VmappedEvalResult(
            candidate=candidate,
            records=[],
            timings=SectionTimings(
                perturbation_s=perturbation_s,
                binding_s=binding_s,
                batch_eval_s=batch_eval_s,
                total_grid_s=total_grid_s,
            ),
            peak_cpu_memory_bytes=_peak_cpu_memory_bytes(),
            peak_cuda_memory_bytes=_peak_cuda_memory_bytes(device),
            metadata={**metadata, "failure_kind": "oom"},
            error=f"{type(exc).__name__}: {exc}",
        )


def _materialize_flat_chunk(
    chunk: Sequence[GridPoint],
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    alphas = torch.tensor(
        [point.alpha for point in chunk],
        device=device,
        dtype=base_vector.dtype,
    )
    betas = torch.tensor(
        [point.beta for point in chunk],
        device=device,
        dtype=base_vector.dtype,
    )
    return (
        base_vector.unsqueeze(0)
        + alphas.unsqueeze(1) * direction_a.unsqueeze(0)
        + betas.unsqueeze(1) * direction_b.unsqueeze(0)
    )


def _loss_for_params(
    model: torch.nn.Module,
    buffers: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    logits_or_predictions = functional_call(model, (params, buffers), (inputs,))
    if loss_name == "cross_entropy":
        return F.cross_entropy(logits_or_predictions, targets, reduction="mean")
    if loss_name == "mse":
        predictions = logits_or_predictions.squeeze(-1)
        targets = targets.to(dtype=predictions.dtype)
        diff = predictions - targets
        return (diff * diff).mean()
    raise ValueError(f"unsupported vmapped functional loss: {loss_name}")


def _move_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = batch
    return (
        inputs.to(device, dtype=torch.float32, non_blocking=True),
        targets.to(device, non_blocking=True),
    )


def _move_base_tensors_to_device(
    device: torch.device,
    *tensors: torch.Tensor,
) -> None:
    for tensor in tensors:
        if not _same_device(tensor.device, device):
            raise ValueError(
                "vmapped evaluator expects base and direction tensors on the "
                f"target device {device}, got {tensor.device}"
            )


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    if actual.type != expected.type:
        return False
    return expected.index is None or actual.index == expected.index


def _chunks(
    points: Sequence[GridPoint],
    size: int,
) -> Iterable[Sequence[GridPoint]]:
    for start in range(0, len(points), size):
        yield points[start : start + size]


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_vmap() -> Any:
    vmap = getattr(torch, "vmap", None)
    if vmap is not None:
        return vmap
    return torch.func.vmap


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cuda error: out of memory" in message
        or "cudnn_status_alloc_failed" in message
        or "mps backend out of memory" in message
        or "not enough memory" in message
    )


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        synchronize = getattr(torch.mps, "synchronize", None)
        if synchronize is not None:
            synchronize()


def _synchronize_after_oom(device: torch.device) -> None:
    try:
        _synchronize(device)
    except Exception:
        pass


def _peak_cpu_memory_bytes() -> int | None:
    snapshot = process_memory_snapshot()
    return snapshot.rss_bytes if snapshot.available else None


def _peak_cuda_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    snapshot = cuda_memory_snapshot(device)
    return snapshot.max_allocated_bytes if snapshot.available else None
