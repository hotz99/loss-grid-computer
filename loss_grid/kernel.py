from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random
import time
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.nn.utils import vector_to_parameters

from loss_grid.config import ExperimentConfig
from loss_grid.data import build_dataloader
from loss_grid.directions import build_direction_vectors
from loss_grid.environment import capture_environment
from loss_grid.grid import build_grid_points
from loss_grid.models import build_model
from loss_grid.resnet20_compiled import build_resnet20_compiled_chunk_evaluator


def maybe_preload_batches(
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    enabled: bool,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    if not enabled or device.type == "cpu":
        return data_loader

    preloaded_batches = []
    preload_start = time.perf_counter()
    for inputs, targets in data_loader:
        inputs_device = inputs.to(device, non_blocking=False)
        targets_device = targets.to(device, non_blocking=False)
        preloaded_batches.append((inputs_device, targets_device))
    preload_s = time.perf_counter() - preload_start
    print(
        "[preload] "
        f"device={device.type} "
        f"batches={len(preloaded_batches)} "
        f"seconds={preload_s:.6f}"
    )
    return preloaded_batches


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def precision_context(precision: str, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def resolve_batch_size(config: ExperimentConfig, device: torch.device) -> int:
    if device.type == "cpu" and config.data.cpu_batch_size is not None:
        return int(config.data.cpu_batch_size)
    if device.type != "cpu" and config.data.gpu_batch_size is not None:
        return int(config.data.gpu_batch_size)
    return int(config.data.batch_size)


def apply_parameter_vector(
    model: torch.nn.Module, parameter_vector: torch.Tensor
) -> None:
    vector_to_parameters(parameter_vector, model.parameters())


def evaluate_loss(
    model: torch.nn.Module,
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    precision: str,
    num_batches: Optional[int],
) -> Tuple[float, float]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    batch_count = 0
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = None
        end_event = None

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(data_loader):
            if num_batches is not None and batch_index >= num_batches:
                break
            if inputs.device != device:
                inputs = inputs.to(device, non_blocking=True)
            if targets.device != device:
                targets = targets.to(device, non_blocking=True)
            with precision_context(precision, device):
                logits = model(inputs)
                loss = loss_fn(logits, targets)
            total_loss += float(loss.detach().cpu())
            batch_count += 1

    gpu_kernel_s = 0.0
    if device.type == "cuda" and start_event is not None and end_event is not None:
        end_event.record()
        torch.cuda.synchronize(device)
        gpu_kernel_s = float(start_event.elapsed_time(end_event) / 1000.0)

    average_loss = total_loss / max(1, batch_count)
    return average_loss, gpu_kernel_s


def evaluate_loss_compiled_chunk(
    compiled_chunk_evaluator: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    precision: str,
    num_batches: Optional[int],
    perturbations: torch.Tensor,
    active_count: int,
) -> Tuple[list[float], float]:
    total_losses = torch.zeros(perturbations.shape[0], device=device)
    batch_count = 0
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = None
        end_event = None

    with torch.no_grad(), torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(data_loader):
            if num_batches is not None and batch_index >= num_batches:
                break
            if inputs.device != device:
                inputs = inputs.to(device, non_blocking=True)
            if targets.device != device:
                targets = targets.to(device, non_blocking=True)
            with precision_context(precision, device):
                total_losses += compiled_chunk_evaluator(inputs, targets, perturbations)
            batch_count += 1

    gpu_kernel_s = 0.0
    if device.type == "cuda" and start_event is not None and end_event is not None:
        end_event.record()
        torch.cuda.synchronize(device)
        gpu_kernel_s = float(start_event.elapsed_time(end_event) / 1000.0)

    average_losses = (total_losses / max(1, batch_count)).detach().cpu().tolist()
    return [float(loss) for loss in average_losses[:active_count]], gpu_kernel_s


@dataclass
class ExecutionContext:
    config: ExperimentConfig
    model: torch.nn.Module
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    device: torch.device
    base_vector_cpu: torch.Tensor
    direction_a_cpu: torch.Tensor
    direction_b_cpu: torch.Tensor
    points: list
    environment: dict
    parameter_names: tuple[str, ...]
    parameter_numels: tuple[int, ...]
    parameter_shapes: tuple[torch.Size, ...]
    buffers: dict[str, torch.Tensor]
    compiled_gpu_chunk_eval_enabled: bool
    compiled_gpu_chunk_size: int
    compiled_gpu_chunk_eval_available: bool
    compiled_chunk_evaluator: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ]


def build_execution_context(
    config: ExperimentConfig,
    device_override: Optional[str] = None,
    capture_env_info: bool = True,
) -> ExecutionContext:
    set_determinism(config.seed)
    model = build_model(config.model)
    device = resolve_device(device_override or config.runtime.device)
    batch_size = resolve_batch_size(config, device)
    data_loader = build_dataloader(config.data, batch_size_override=batch_size)
    data_loader = maybe_preload_batches(
        data_loader=data_loader,
        device=device,
        enabled=bool(config.runtime.preload_gpu_batches),
    )
    base_vector_cpu, direction_a_cpu, direction_b_cpu = build_direction_vectors(
        model, config.seed
    )
    points = build_grid_points(config.grid)
    dataset_size = len(data_loader.dataset) if hasattr(data_loader, "dataset") else "n/a"
    loader_batch_size = getattr(data_loader, "batch_size", "n/a")
    use_compiled_gpu_chunk_eval = bool(config.runtime.compile_gpu_chunk_eval)
    print(
        "[run] "
        f"model={config.model.name} "
        f"device={device.type} "
        f"use_compiled_gpu_chunk_eval={use_compiled_gpu_chunk_eval} "
        f"dataset_size={dataset_size} "
        f"batch_size={loader_batch_size} "
        f"num_batches={config.runtime.num_batches}"
    )
    model = model.to(device)
    apply_parameter_vector(model, base_vector_cpu.to(device))
    parameter_names = tuple(name for name, _ in model.named_parameters())
    parameter_numels = tuple(
        parameter.numel() for _, parameter in model.named_parameters()
    )
    parameter_shapes = tuple(
        parameter.shape for _, parameter in model.named_parameters()
    )
    buffers = {name: buffer for name, buffer in model.named_buffers()}
    compiled_chunk_evaluator = None
    compiled_gpu_chunk_eval_available = False
    if (
        device.type != "cpu"
        and use_compiled_gpu_chunk_eval
        and len(data_loader) > 0
        and isinstance(data_loader, list)
    ):
        example_inputs, example_targets = data_loader[0]
        compiled_chunk_evaluator = build_resnet20_compiled_chunk_evaluator(
            model_name=config.model.name,
            use_skip=bool(getattr(model, "use_skip", True)),
            base_vector=base_vector_cpu.to(device),
            direction_a=direction_a_cpu.to(device),
            direction_b=direction_b_cpu.to(device),
            parameter_numels=parameter_numels,
            parameter_shapes=parameter_shapes,
            buffers=buffers,
            example_inputs=example_inputs,
            example_targets=example_targets,
            chunk_size=max(1, int(config.runtime.compile_gpu_chunk_size)),
            device=device,
        )
        compiled_gpu_chunk_eval_available = compiled_chunk_evaluator is not None
    
    return ExecutionContext(
        config=config,
        model=model,
        data_loader=data_loader,
        device=device,
        base_vector_cpu=base_vector_cpu,
        direction_a_cpu=direction_a_cpu,
        direction_b_cpu=direction_b_cpu,
        points=points,
        environment=capture_environment() if capture_env_info else {},
        parameter_names=parameter_names,
        parameter_numels=parameter_numels,
        parameter_shapes=parameter_shapes,
        buffers=buffers,
        compiled_gpu_chunk_eval_enabled=use_compiled_gpu_chunk_eval,
        compiled_gpu_chunk_size=max(1, int(config.runtime.compile_gpu_chunk_size)),
        compiled_gpu_chunk_eval_available=compiled_gpu_chunk_eval_available,
        compiled_chunk_evaluator=compiled_chunk_evaluator,
    )


def build_parameter_vector(
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    return base_vector + (alpha * direction_a) + (beta * direction_b)
