from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random
import time
from typing import List, Sequence, TypeAlias

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from src.system_schema import (
    GridSpec,
    HybridMode,
    SchedulerRequest,
)
from src.results import synchronize_device
from src.workloads import WORKLOADS

# ------------------------------
#              TYPES
# ------------------------------


@dataclass(frozen=True, slots=True)
class GridPoint:
    linear_idx: int
    row: int
    col: int
    alpha: float
    beta: float


Surface: TypeAlias = list[tuple[int, int, float]]


# ------------------------------
#           COMMON HELPERS
# ------------------------------


def build_grid_points(config: GridSpec):
    alphas = torch.linspace(-config.scale, config.scale, config.resolution).tolist()
    betas = torch.linspace(-config.scale, config.scale, config.resolution).tolist()
    points: List[GridPoint] = []
    linear_idx = 0
    for row, alpha in enumerate(alphas):
        for col, beta in enumerate(betas):
            points.append(GridPoint(linear_idx, row, col, alpha, beta))
            linear_idx += 1
    return points


def resolve_device(device: str = "auto") -> torch.device:
    return torch.device(
        device
        if device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )


def build_output_dir(
    request: SchedulerRequest,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path("outputs") / f"{request.task.name}-{request.mode._tag}-{timestamp}")


def prepare_model_and_data(
    request: SchedulerRequest,
    device: torch.device,
    seed: int,
):
    definition = WORKLOADS[request.task.name]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
    ):
        torch.mps.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = definition.build_model(request.task).float()
    batch_size = (
        request.mode.cpu_batch_size
        if device.type == "cpu" and isinstance(request.mode, HybridMode)
        else (
            request.mode.gpu_batch_size
        )
    )
    dataset = definition.build_dataset(request.task, seed)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    preload_s = 0.0
    print(
        "[run] "
        f"workload={request.task.name} "
        f"model={request.task.model} "
        f"device={device.type} "
        f"total_samples={len(dataset)} "
        f"batch_size={batch_size}"
    )

    model = model.to(device)
    base_vector_cpu, direction_a_cpu, direction_b_cpu = build_direction_vectors(
        model,
        seed,
    )
    return (
        model,
        data_loader,
        preload_s,
        base_vector_cpu,
        direction_a_cpu,
        direction_b_cpu,
    )


def throughput(points: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(points) / float(seconds)


def apply_gpu_slowdown(
    device: torch.device,
    gpu_slowdown_factor: float,
    elapsed_s: float,
):
    if device.type == "cpu":
        return

    synchronize_device(device)
    extra_delay = elapsed_s * (gpu_slowdown_factor - 1.0)
    time.sleep(extra_delay)


# ------------------------------
#           CORE ALGORITHM STEPS
# ------------------------------


def _normalize_filterwise(
    parameter: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    if parameter.ndim <= 1:
        param_norm = torch.linalg.vector_norm(parameter.reshape(-1))
        dir_norm = torch.linalg.vector_norm(direction.reshape(-1))
        if float(dir_norm) == 0.0:
            return direction
        scale = (
            (param_norm / dir_norm) if float(param_norm) > 0.0 else torch.tensor(1.0)
        )
        return direction * scale.to(direction.dtype)

    flattened_param = parameter.reshape(parameter.shape[0], -1)
    flattened_dir = direction.reshape(direction.shape[0], -1)
    param_norms = torch.linalg.vector_norm(flattened_param, dim=1, keepdim=True)
    dir_norms = torch.linalg.vector_norm(flattened_dir, dim=1, keepdim=True)
    dir_norms = torch.where(dir_norms == 0, torch.ones_like(dir_norms), dir_norms)
    scales = torch.where(
        param_norms > 0, param_norms / dir_norms, torch.ones_like(param_norms)
    )
    return (flattened_dir * scales).reshape_as(parameter)


def build_direction_vectors(model: torch.nn.Module, seed: int):
    generator = torch.Generator().manual_seed(seed)
    params = [
        param.detach().cpu().to(torch.float32).clone() for param in model.parameters()
    ]
    directions_a = []
    directions_b = []

    for parameter in params:
        rand_a = torch.randn(
            parameter.shape, generator=generator, dtype=parameter.dtype
        )
        rand_b = torch.randn(
            parameter.shape, generator=generator, dtype=parameter.dtype
        )
        directions_a.append(_normalize_filterwise(parameter, rand_a))
        directions_b.append(_normalize_filterwise(parameter, rand_b))

    base = parameters_to_vector(params).detach().cpu()
    vec_a = parameters_to_vector(directions_a).detach().cpu().to(torch.float32)
    vec_b = parameters_to_vector(directions_b).detach().cpu().to(torch.float32)
    return base, vec_a, vec_b


def evaluate_points_on_device(
    request: SchedulerRequest,
    model,
    data_loader,
    device,
    chunk: Sequence[GridPoint],
    base_vector_device: torch.Tensor,
    direction_a_device: torch.Tensor,
    direction_b_device: torch.Tensor,
):
    records: Surface = []
    definition = WORKLOADS[request.task.name]

    for point in chunk:
        perturbed_variant = (
            base_vector_device
            + (point.alpha * direction_a_device)
            + (point.beta * direction_b_device)
        )

        vector_to_parameters(perturbed_variant, model.parameters())

        model.eval()
        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in data_loader:
                loss, batch_size = definition.compute_loss(model, batch, device)
                total_loss += float(loss.cpu()) * batch_size
                total_examples += batch_size

        avg_loss = total_loss / max(1, total_examples)
        records.append((point.row, point.col, avg_loss))

    return records
