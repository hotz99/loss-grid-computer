from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.nn.utils import vector_to_parameters

from src.backends.base import (
    GridPoint,
    Surface,
    build_grid_points,
    prepare_model_and_data,
    resolve_device,
)
from src.functional_eval.layout import (
    flat_chunk_to_batched_param_dict,
    flat_vector_to_param_dict,
    make_functional_state,
)
from src.functional_eval.memory import SectionTimings
from src.results import synchronize_device
from src.schemas import SchedulerRequest, VanillaMode


@dataclass(frozen=True)
class CompileDiagnostics:
    graph_break_count: int | None
    recompile_count: int | None
    counters: dict[str, dict[str, int]]


@dataclass(frozen=True)
class CompiledEvalResult:
    candidate: str
    records: Surface
    timings: SectionTimings
    compile_s: float
    first_call_s: float
    steady_grid_s: float
    graph_break_count: int | None
    recompile_count: int | None
    compile_counters: dict[str, dict[str, int]]
    metadata: dict[str, Any]
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def run_compiled_forward_surface(
    request: SchedulerRequest,
    *,
    seed: int = 1337,
    compile_mode: str | None = None,
) -> CompiledEvalResult:
    """Compile the model forward/loss path while leaving grid orchestration eager."""
    assert isinstance(request.mode, VanillaMode)
    candidate = "compiled_forward"
    device = resolve_device(request.device)
    started = perf_counter()
    _reset_compile_diagnostics()

    try:
        (
            model,
            data_loader,
            _preload_s,
            base_vector_cpu,
            direction_a_cpu,
            direction_b_cpu,
        ) = prepare_model_and_data(request, device, seed)
        model.eval()
        points = build_grid_points(request.grid)
        base_vector = base_vector_cpu.to(device)
        direction_a = direction_a_cpu.to(device)
        direction_b = direction_b_cpu.to(device)

        def forward_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            outputs = model(inputs)
            return _loss_from_outputs(outputs, targets, request.task.loss)

        compiled_forward_loss = _compile(forward_loss, compile_mode)

        first_call_s = _warm_compiled_forward(
            compiled_forward_loss,
            model,
            data_loader,
            device,
            points,
            base_vector,
            direction_a,
            direction_b,
        )
        records, timings = _evaluate_compiled_forward(
            request,
            compiled_forward_loss,
            model,
            data_loader,
            device,
            points,
            base_vector,
            direction_a,
            direction_b,
        )
        diagnostics = _compile_diagnostics()
        return CompiledEvalResult(
            candidate=candidate,
            records=records,
            timings=timings,
            compile_s=first_call_s,
            first_call_s=first_call_s,
            steady_grid_s=timings.total_grid_s,
            graph_break_count=diagnostics.graph_break_count,
            recompile_count=diagnostics.recompile_count,
            compile_counters=diagnostics.counters,
            metadata={
                "compile_target": "model_forward_loss",
                "device": device.type,
                "workload": request.task.name,
                "grid_resolution": request.grid.resolution,
                "grid_points": len(points),
                "compile_mode": compile_mode or "default",
            },
        )
    except Exception as exc:
        diagnostics = _compile_diagnostics()
        return _error_result(
            candidate,
            started,
            diagnostics,
            {
                "compile_target": "model_forward_loss",
                "device": device.type,
                "workload": request.task.name,
                "compile_mode": compile_mode or "default",
            },
            exc,
        )


def run_compiled_functional_surface(
    request: SchedulerRequest,
    *,
    seed: int = 1337,
    compile_mode: str | None = None,
) -> CompiledEvalResult:
    """Compile a stateless functional_call forward/loss path."""
    assert isinstance(request.mode, VanillaMode)
    candidate = "compiled_functional"
    device = resolve_device(request.device)
    started = perf_counter()
    _reset_compile_diagnostics()

    try:
        (
            model,
            data_loader,
            _preload_s,
            base_vector_cpu,
            direction_a_cpu,
            direction_b_cpu,
        ) = prepare_model_and_data(request, device, seed)
        model.eval()
        points = build_grid_points(request.grid)
        base_vector = base_vector_cpu.to(device)
        direction_a = direction_a_cpu.to(device)
        direction_b = direction_b_cpu.to(device)
        _base_params, buffers, layout = make_functional_state(model)

        def functional_loss(
            params: dict[str, torch.Tensor],
            inputs: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            outputs = functional_call(model, (params, buffers), (inputs,))
            return _loss_from_outputs(outputs, targets, request.task.loss)

        compiled_functional_loss = _compile(functional_loss, compile_mode)

        first_call_s = _warm_compiled_functional(
            compiled_functional_loss,
            data_loader,
            device,
            points,
            base_vector,
            direction_a,
            direction_b,
            layout,
        )
        records, timings = _evaluate_compiled_functional(
            compiled_functional_loss,
            data_loader,
            device,
            points,
            base_vector,
            direction_a,
            direction_b,
            layout,
        )
        diagnostics = _compile_diagnostics()
        return CompiledEvalResult(
            candidate=candidate,
            records=records,
            timings=timings,
            compile_s=first_call_s,
            first_call_s=first_call_s,
            steady_grid_s=timings.total_grid_s,
            graph_break_count=diagnostics.graph_break_count,
            recompile_count=diagnostics.recompile_count,
            compile_counters=diagnostics.counters,
            metadata={
                "compile_target": "functional_call_forward_loss",
                "device": device.type,
                "workload": request.task.name,
                "grid_resolution": request.grid.resolution,
                "grid_points": len(points),
                "buffer_count": len(buffers),
                "parameter_count": len(layout.entries),
                "compile_mode": compile_mode or "default",
            },
        )
    except Exception as exc:
        diagnostics = _compile_diagnostics()
        return _error_result(
            candidate,
            started,
            diagnostics,
            {
                "compile_target": "functional_call_forward_loss",
                "device": device.type,
                "workload": request.task.name,
                "compile_mode": compile_mode or "default",
            },
            exc,
        )


def run_compiled_vmapped_surface(
    request: SchedulerRequest,
    *,
    seed: int = 1337,
    point_chunk_size: int,
    compile_mode: str | None = None,
) -> CompiledEvalResult:
    """Compile the vmapped chunk forward/loss path."""
    assert isinstance(request.mode, VanillaMode)
    if point_chunk_size < 1:
        raise ValueError(f"point_chunk_size must be >= 1, got {point_chunk_size}")

    candidate = f"compiled_vmapped_chunk_{point_chunk_size}"
    device = resolve_device(request.device)
    started = perf_counter()
    _reset_compile_diagnostics()

    try:
        (
            model,
            data_loader,
            _preload_s,
            base_vector_cpu,
            direction_a_cpu,
            direction_b_cpu,
        ) = prepare_model_and_data(request, device, seed)
        model.eval()
        points = build_grid_points(request.grid)
        base_vector = base_vector_cpu.to(device)
        direction_a = direction_a_cpu.to(device)
        direction_b = direction_b_cpu.to(device)
        _base_params, buffers, layout = make_functional_state(model)
        vmap = _resolve_vmap()

        def vmapped_loss(
            batched_params: dict[str, torch.Tensor],
            inputs: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            return vmap(
                lambda params: _loss_for_params(
                    model,
                    buffers,
                    params,
                    inputs,
                    targets,
                    request.task.loss,
                ),
                randomness="error",
            )(batched_params)

        compiled_vmapped_loss = _compile(vmapped_loss, compile_mode)

        first_call_s = _warm_compiled_vmapped(
            compiled_vmapped_loss,
            data_loader,
            device,
            points,
            base_vector,
            direction_a,
            direction_b,
            layout,
            point_chunk_size,
        )
        records, timings = _evaluate_compiled_vmapped(
            compiled_vmapped_loss,
            data_loader,
            device,
            points,
            base_vector,
            direction_a,
            direction_b,
            layout,
            point_chunk_size,
        )
        diagnostics = _compile_diagnostics()
        return CompiledEvalResult(
            candidate=candidate,
            records=records,
            timings=timings,
            compile_s=first_call_s,
            first_call_s=first_call_s,
            steady_grid_s=timings.total_grid_s,
            graph_break_count=diagnostics.graph_break_count,
            recompile_count=diagnostics.recompile_count,
            compile_counters=diagnostics.counters,
            metadata={
                "compile_target": "vmapped_functional_chunk_forward_loss",
                "device": device.type,
                "workload": request.task.name,
                "grid_resolution": request.grid.resolution,
                "grid_points": len(points),
                "point_chunk_size": point_chunk_size,
                "buffer_count": len(buffers),
                "parameter_count": len(layout.entries),
                "compile_mode": compile_mode or "default",
            },
        )
    except Exception as exc:
        diagnostics = _compile_diagnostics()
        return _error_result(
            candidate,
            started,
            diagnostics,
            {
                "compile_target": "vmapped_functional_chunk_forward_loss",
                "device": device.type,
                "workload": request.task.name,
                "point_chunk_size": point_chunk_size,
                "compile_mode": compile_mode or "default",
            },
            exc,
        )


def _evaluate_compiled_forward(
    request: SchedulerRequest,
    compiled_forward_loss,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    points: Sequence[GridPoint],
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
) -> tuple[Surface, SectionTimings]:
    records: Surface = []
    perturbation_s = 0.0
    binding_s = 0.0
    batch_eval_s = 0.0
    synchronize_device(device)
    total_started = perf_counter()

    with torch.no_grad():
        for point in points:
            perturbation_started = perf_counter()
            perturbed_variant = _perturbed_vector(
                point,
                base_vector,
                direction_a,
                direction_b,
            )
            perturbation_s += _elapsed_since(perturbation_started, device)

            binding_started = perf_counter()
            vector_to_parameters(perturbed_variant, model.parameters())
            model.eval()
            binding_s += _elapsed_since(binding_started, device)

            total_loss = 0.0
            total_examples = 0
            batch_eval_started = perf_counter()
            for batch in data_loader:
                inputs, targets = _move_batch(batch, device, request.task.loss)
                loss = compiled_forward_loss(inputs, targets)
                batch_size = int(targets.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size
            batch_eval_s += _elapsed_since(batch_eval_started, device)

            records.append((point.row, point.col, total_loss / max(1, total_examples)))

    total_grid_s = _elapsed_since(total_started, device)
    return records, SectionTimings(
        perturbation_s=perturbation_s,
        binding_s=binding_s,
        batch_eval_s=batch_eval_s,
        total_grid_s=total_grid_s,
    )


def _evaluate_compiled_functional(
    compiled_functional_loss,
    data_loader,
    device: torch.device,
    points: Sequence[GridPoint],
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    layout,
) -> tuple[Surface, SectionTimings]:
    records: Surface = []
    perturbation_s = 0.0
    binding_s = 0.0
    batch_eval_s = 0.0
    synchronize_device(device)
    total_started = perf_counter()

    with torch.no_grad():
        for point in points:
            perturbation_started = perf_counter()
            perturbed_variant = _perturbed_vector(
                point,
                base_vector,
                direction_a,
                direction_b,
            )
            perturbation_s += _elapsed_since(perturbation_started, device)

            binding_started = perf_counter()
            params = flat_vector_to_param_dict(perturbed_variant, layout)
            binding_s += _elapsed_since(binding_started, device)

            total_loss = 0.0
            total_examples = 0
            batch_eval_started = perf_counter()
            for batch in data_loader:
                inputs, targets = _move_batch(batch, device, "")
                loss = compiled_functional_loss(params, inputs, targets)
                batch_size = int(targets.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size
            batch_eval_s += _elapsed_since(batch_eval_started, device)

            records.append((point.row, point.col, total_loss / max(1, total_examples)))

    total_grid_s = _elapsed_since(total_started, device)
    return records, SectionTimings(
        perturbation_s=perturbation_s,
        binding_s=binding_s,
        batch_eval_s=batch_eval_s,
        total_grid_s=total_grid_s,
    )


def _evaluate_compiled_vmapped(
    compiled_vmapped_loss,
    data_loader,
    device: torch.device,
    points: Sequence[GridPoint],
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    layout,
    point_chunk_size: int,
) -> tuple[Surface, SectionTimings]:
    records: Surface = []
    perturbation_s = 0.0
    binding_s = 0.0
    batch_eval_s = 0.0
    synchronize_device(device)
    total_started = perf_counter()

    with torch.no_grad():
        for chunk in _chunks(points, point_chunk_size):
            perturbation_started = perf_counter()
            flat_vectors = _materialize_flat_chunk(
                chunk,
                base_vector,
                direction_a,
                direction_b,
                device,
            )
            perturbation_s += _elapsed_since(perturbation_started, device)

            binding_started = perf_counter()
            batched_parameters = flat_chunk_to_batched_param_dict(flat_vectors, layout)
            binding_s += _elapsed_since(binding_started, device)

            weighted_loss_sum = torch.zeros(
                len(chunk),
                device=device,
                dtype=torch.float32,
            )
            total_examples = 0
            batch_eval_started = perf_counter()
            for batch in data_loader:
                inputs, targets = _move_batch(batch, device, "")
                losses = compiled_vmapped_loss(batched_parameters, inputs, targets)
                batch_size = int(targets.shape[0])
                weighted_loss_sum += losses.detach().to(torch.float32) * batch_size
                total_examples += batch_size
            batch_eval_s += _elapsed_since(batch_eval_started, device)

            averages = (weighted_loss_sum / max(1, total_examples)).detach().cpu().tolist()
            for point, avg_loss in zip(chunk, averages):
                records.append((point.row, point.col, float(avg_loss)))

    total_grid_s = _elapsed_since(total_started, device)
    return records, SectionTimings(
        perturbation_s=perturbation_s,
        binding_s=binding_s,
        batch_eval_s=batch_eval_s,
        total_grid_s=total_grid_s,
    )


def _warm_compiled_forward(
    compiled_forward_loss,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    points: Sequence[GridPoint],
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
) -> float:
    point = points[0]
    batch = next(iter(data_loader))
    model.eval()
    vector_to_parameters(
        _perturbed_vector(point, base_vector, direction_a, direction_b),
        model.parameters(),
    )
    inputs, targets = _move_batch(batch, device, "")
    synchronize_device(device)
    started = perf_counter()
    with torch.no_grad():
        _ = compiled_forward_loss(inputs, targets)
    return _elapsed_since(started, device)


def _warm_compiled_functional(
    compiled_functional_loss,
    data_loader,
    device: torch.device,
    points: Sequence[GridPoint],
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    layout,
) -> float:
    point = points[0]
    batch = next(iter(data_loader))
    params = flat_vector_to_param_dict(
        _perturbed_vector(point, base_vector, direction_a, direction_b),
        layout,
    )
    inputs, targets = _move_batch(batch, device, "")
    synchronize_device(device)
    started = perf_counter()
    with torch.no_grad():
        _ = compiled_functional_loss(params, inputs, targets)
    return _elapsed_since(started, device)


def _warm_compiled_vmapped(
    compiled_vmapped_loss,
    data_loader,
    device: torch.device,
    points: Sequence[GridPoint],
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    layout,
    point_chunk_size: int,
) -> float:
    chunk = list(points[:point_chunk_size])
    batch = next(iter(data_loader))
    flat_vectors = _materialize_flat_chunk(
        chunk,
        base_vector,
        direction_a,
        direction_b,
        device,
    )
    batched_parameters = flat_chunk_to_batched_param_dict(flat_vectors, layout)
    inputs, targets = _move_batch(batch, device, "")
    synchronize_device(device)
    started = perf_counter()
    with torch.no_grad():
        _ = compiled_vmapped_loss(batched_parameters, inputs, targets)
    return _elapsed_since(started, device)


def _loss_from_outputs(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    if loss_name == "cross_entropy":
        return F.cross_entropy(outputs, targets, reduction="mean")
    if loss_name == "mse":
        predictions = outputs.squeeze(-1)
        targets = targets.to(dtype=predictions.dtype)
        return F.mse_loss(predictions, targets, reduction="mean")
    raise ValueError(f"unsupported loss for compilation MVP: {loss_name}")


def _loss_for_params(
    model: torch.nn.Module,
    buffers: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    outputs = functional_call(model, (params, buffers), (inputs,))
    return _loss_from_outputs(outputs, targets, loss_name)


def _move_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    del loss_name
    inputs, targets = batch
    return (
        inputs.to(device, dtype=torch.float32, non_blocking=True),
        targets.to(device, non_blocking=True),
    )


def _perturbed_vector(
    point: GridPoint,
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
) -> torch.Tensor:
    return (
        base_vector
        + (point.alpha * direction_a)
        + (point.beta * direction_b)
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


def _chunks(
    points: Sequence[GridPoint],
    size: int,
) -> Iterable[Sequence[GridPoint]]:
    for start in range(0, len(points), size):
        yield points[start : start + size]


def _elapsed_since(started_at_s: float, device: torch.device) -> float:
    synchronize_device(device)
    return perf_counter() - started_at_s


def _resolve_vmap() -> Any:
    vmap = getattr(torch, "vmap", None)
    if vmap is not None:
        return vmap
    return torch.func.vmap


def _compile(fn, compile_mode: str | None):
    if compile_mode is None:
        return torch.compile(fn)
    return torch.compile(fn, mode=compile_mode)


def _reset_compile_diagnostics() -> None:
    try:
        import torch._dynamo as dynamo

        dynamo.reset()
    except Exception:
        pass
    try:
        from torch._dynamo.utils import counters

        counters.clear()
    except Exception:
        pass


def _compile_diagnostics() -> CompileDiagnostics:
    counters = _read_compile_counters()
    graph_break_count = _sum_counter(counters, "graph_break")
    recompile_count = (
        _sum_counter(counters, "recompiles")
        + _sum_counter(counters, "recompile")
    )
    return CompileDiagnostics(
        graph_break_count=graph_break_count,
        recompile_count=recompile_count,
        counters=counters,
    )


def _read_compile_counters() -> dict[str, dict[str, int]]:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}

    output: dict[str, dict[str, int]] = {}
    for group, counter in counters.items():
        if not counter:
            continue
        output[str(group)] = {
            str(key): int(value)
            for key, value in counter.items()
            if isinstance(value, int)
        }
    return output


def _sum_counter(counters: dict[str, dict[str, int]], group: str) -> int | None:
    if group not in counters:
        return 0
    return int(sum(counters[group].values()))


def _error_result(
    candidate: str,
    started: float,
    diagnostics: CompileDiagnostics,
    metadata: dict[str, Any],
    exc: Exception,
) -> CompiledEvalResult:
    elapsed = perf_counter() - started
    return CompiledEvalResult(
        candidate=candidate,
        records=[],
        timings=SectionTimings(total_grid_s=elapsed),
        compile_s=0.0,
        first_call_s=0.0,
        steady_grid_s=elapsed,
        graph_break_count=diagnostics.graph_break_count,
        recompile_count=diagnostics.recompile_count,
        compile_counters=diagnostics.counters,
        metadata={**metadata, "failure_kind": type(exc).__name__},
        error=f"{type(exc).__name__}: {exc}",
    )
