from __future__ import annotations

from typing import Sequence

import torch
from torch.nn.utils import vector_to_parameters

from src.backends.base import (
    GridPoint,
    Surface,
    build_grid_points,
    prepare_model_and_data,
)
from src.system_schema import SchedulerRequest, VanillaMode
from src.results import synchronize_device


def _prepare_reference_model_and_data(
    request: SchedulerRequest,
    device: torch.device,
    *,
    seed: int,
):
    if request.task.name != "cifar10_resnet20_classification":
        raise ValueError(
            "original_algo only supports cifar10_resnet20_classification"
        )
    (
        model,
        data_loader,
        _preload_s,
        base_vector_cpu,
        direction_a_cpu,
        direction_b_cpu,
    ) = prepare_model_and_data(
        request,
        device,
        seed=seed,
    )
    return model, data_loader, base_vector_cpu, direction_a_cpu, direction_b_cpu


def _evaluate_points_reference(
    model,
    data_loader,
    device,
    chunk: Sequence[GridPoint],
    base_vector_device: torch.Tensor,
    direction_a_device: torch.Tensor,
    direction_b_device: torch.Tensor,
) -> Surface:
    records: Surface = []
    loss_fn = torch.nn.CrossEntropyLoss()

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
            for inputs, targets in data_loader:
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                batch_size = int(targets.shape[0])
                total_loss += float(loss.cpu()) * batch_size
                total_examples += batch_size

        avg_loss = total_loss / max(1, total_examples)
        records.append((point.row, point.col, avg_loss))

    return records


def run_reference_surface(
    request: SchedulerRequest,
    *,
    seed: int = 1337,
) -> Surface:
    assert isinstance(request.mode, VanillaMode)
    torch_device = torch.device(
        request.device
        if request.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    (
        model,
        data_loader,
        base_vector_cpu,
        direction_a_cpu,
        direction_b_cpu,
    ) = _prepare_reference_model_and_data(
        request,
        torch_device,
        seed=seed,
    )

    points = build_grid_points(request.grid)
    base_vector_device = base_vector_cpu.to(torch_device)
    direction_a_device = direction_a_cpu.to(torch_device)
    direction_b_device = direction_b_cpu.to(torch_device)
    vector_to_parameters(base_vector_device, model.parameters())
    synchronize_device(torch_device)

    return _evaluate_points_reference(
        model=model,
        data_loader=data_loader,
        device=torch_device,
        chunk=points,
        base_vector_device=base_vector_device,
        direction_a_device=direction_a_device,
        direction_b_device=direction_b_device,
    )
