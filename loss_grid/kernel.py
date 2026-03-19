from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random
import time
from typing import Iterable, Optional, Tuple

import torch
from torch.nn.utils import vector_to_parameters

from loss_grid.config import ExperimentConfig
from loss_grid.data import build_dataloader
from loss_grid.directions import build_direction_vectors
from loss_grid.environment import capture_environment
from loss_grid.grid import build_grid_points
from loss_grid.models import build_model


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
            inputs = inputs.to(device, non_blocking=True)
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
    base_vector_cpu, direction_a_cpu, direction_b_cpu = build_direction_vectors(
        model, config.seed
    )
    points = build_grid_points(config.grid)
    dataset_size = len(data_loader.dataset) if hasattr(data_loader, "dataset") else "n/a"
    loader_batch_size = getattr(data_loader, "batch_size", "n/a")
    print(
        "[run] "
        f"model={config.model.name} "
        f"device={device.type} "
        f"dataset_size={dataset_size} "
        f"batch_size={loader_batch_size} "
        f"num_batches={config.runtime.num_batches}"
    )
    model = model.to(device)
    apply_parameter_vector(model, base_vector_cpu.to(device))
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
    )


def build_parameter_vector(
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    return base_vector + (alpha * direction_a) + (beta * direction_b)
