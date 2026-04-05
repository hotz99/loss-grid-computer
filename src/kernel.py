from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data import Cifar10Dataset, build_dataloader
from src.directions import build_direction_vectors
from src.grid import build_grid_points
from src.models import build_model, build_resnet20_compiled_chunk_evaluator


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


def resolve_device(device_override: Optional[str] = None) -> torch.device:
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_batch_size(config: ExperimentConfig, device: torch.device) -> int:
    if device.type == "cpu" and config.data.cpu_batch_size is not None:
        return config.data.cpu_batch_size
    if device.type != "cpu" and config.data.gpu_batch_size is not None:
        return config.data.gpu_batch_size
    return config.data.batch_size


def evaluate_loss(
    model: torch.nn.Module,
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_batches: Optional[int],
) -> float:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(data_loader):
            if num_batches is not None and batch_index >= num_batches:
                break
            if inputs.device != device:
                inputs = inputs.to(device, non_blocking=True)
            if targets.device != device:
                targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            total_loss += float(loss.detach().cpu())
            batch_count += 1

    average_loss = total_loss / max(1, batch_count)
    return average_loss


def evaluate_loss_compiled_chunk(
    compiled_chunk_evaluator: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_batches: Optional[int],
    perturbations: torch.Tensor,
    active_count: int,
) -> list[float]:
    total_losses = torch.zeros(perturbations.shape[0], device=device)
    batch_count = 0

    with torch.no_grad(), torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(data_loader):
            if num_batches is not None and batch_index >= num_batches:
                break
            if inputs.device != device:
                inputs = inputs.to(device, non_blocking=True)
            if targets.device != device:
                targets = targets.to(device, non_blocking=True)
            total_losses += compiled_chunk_evaluator(inputs, targets, perturbations)
            batch_count += 1

    average_losses = (total_losses / max(1, batch_count)).detach().cpu().tolist()
    return [float(loss) for loss in average_losses[:active_count]]


@dataclass
class ExecutionContext:
    config: ExperimentConfig
    model: torch.nn.Module
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    device: torch.device
    preload_s: float
    compile_s: float
    base_vector_cpu: torch.Tensor
    direction_a_cpu: torch.Tensor
    direction_b_cpu: torch.Tensor
    points: list
    compiled_gpu_chunk_eval_enabled: bool
    compiled_gpu_chunk_size: int
    compiled_chunk_evaluator: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ]


def _resolve_compile_gpu_chunk_size(config: ExperimentConfig, total_points: int) -> int:
    configured = config.runtime.compile_gpu_chunk_size
    if configured is not None:
        return configured

    fixed_gpu_chunk_size = config.decomposition.fixed_gpu_chunk_size
    if fixed_gpu_chunk_size is not None:
        return min(
            total_points,
            config.decomposition.gpu_chunk_size_max,
            fixed_gpu_chunk_size,
        )

    gpu_initial_ratio = config.decomposition.gpu_initial_ratio
    return min(total_points, max(1, int(round(total_points * gpu_initial_ratio))))


def prepare_model_and_data(
    config: ExperimentConfig,
    device_override: Optional[str] = None,
    capture_env_info: bool = True,
) -> tuple[
    torch.nn.Module,
    torch.device,
    Iterable[Tuple[torch.Tensor, torch.Tensor]],
    float,
]:
    set_determinism(config.seed)
    model = build_model(config.model)
    device = resolve_device(device_override)
    batch_size = resolve_batch_size(config, device)
    data_loader: DataLoader[Cifar10Dataset] = build_dataloader(
        config.data, batch_size_override=batch_size
    )
    dataset_size = len(data_loader.dataset)
    loader_batch_size = data_loader.batch_size
    preload_s = 0.0
    if config.runtime.preload_gpu_batches and device.type != "cpu":
        preloaded_batches = []
        preload_start = time.perf_counter()
        for batch_index, (inputs, targets) in enumerate(data_loader):
            if (
                config.runtime.preload_max_batches is not None
                and batch_index >= config.runtime.preload_max_batches
            ):
                break
            inputs_device = inputs.to(device, non_blocking=False)
            targets_device = targets.to(device, non_blocking=False)
            preloaded_batches.append((inputs_device, targets_device))
        preload_s = time.perf_counter() - preload_start
        print(
            "[preload] "
            f"device={device.type} "
            f"batches={len(preloaded_batches)} "
            f"max_batches={config.runtime.preload_max_batches} "
            f"seconds={preload_s:.6f}"
        )
        data_loader = preloaded_batches
    use_compiled_gpu_chunk_eval = config.runtime.compile_gpu_chunk_eval
    if capture_env_info:
        print(
            "[run] "
            "model=resnet20 "
            f"device={device.type} "
            f"use_compiled_gpu_chunk_eval={use_compiled_gpu_chunk_eval} "
            f"dataset_size={dataset_size} "
            f"batch_size={loader_batch_size} "
            f"num_batches={config.runtime.num_batches}"
        )
    model = model.to(device)
    return model, device, data_loader, preload_s


def build_execution_context(
    config: ExperimentConfig,
    device_override: Optional[str] = None,
    capture_env_info: bool = True,
) -> ExecutionContext:
    model, device, data_loader, preload_s = prepare_model_and_data(
        config=config,
        device_override=device_override,
        capture_env_info=capture_env_info,
    )
    base_vector_cpu, direction_a_cpu, direction_b_cpu = build_direction_vectors(
        model, config.seed
    )
    points = build_grid_points(config.grid)
    vector_to_parameters(base_vector_cpu.to(device), model.parameters())
    use_compiled_gpu_chunk_eval = config.runtime.compile_gpu_chunk_eval
    resolved_compile_gpu_chunk_size = _resolve_compile_gpu_chunk_size(
        config=config,
        total_points=len(points),
    )
    return ExecutionContext(
        config=config,
        model=model,
        data_loader=data_loader,
        device=device,
        preload_s=preload_s,
        compile_s=0.0,
        base_vector_cpu=base_vector_cpu,
        direction_a_cpu=direction_a_cpu,
        direction_b_cpu=direction_b_cpu,
        points=points,
        compiled_gpu_chunk_eval_enabled=use_compiled_gpu_chunk_eval,
        compiled_gpu_chunk_size=resolved_compile_gpu_chunk_size,
        compiled_chunk_evaluator=None,
    )


def compile_chunk_evaluator(context: ExecutionContext) -> float:
    parameter_numels = tuple(
        parameter.numel() for _, parameter in context.model.named_parameters()
    )
    parameter_shapes = tuple(
        parameter.shape for _, parameter in context.model.named_parameters()
    )
    buffers = {name: buffer for name, buffer in context.model.named_buffers()}
    example_inputs, example_targets = context.data_loader[0]

    compile_start = time.perf_counter()
    context.compiled_chunk_evaluator = build_resnet20_compiled_chunk_evaluator(
        model_name="resnet20",
        base_vector=context.base_vector_cpu.to(context.device),
        direction_a=context.direction_a_cpu.to(context.device),
        direction_b=context.direction_b_cpu.to(context.device),
        parameter_numels=parameter_numels,
        parameter_shapes=parameter_shapes,
        buffers=buffers,
        example_inputs=example_inputs,
        example_targets=example_targets,
        chunk_size=context.compiled_gpu_chunk_size,
        device=context.device,
    )
    context.compile_s = time.perf_counter() - compile_start
    return context.compile_s
