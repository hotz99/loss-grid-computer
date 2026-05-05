from __future__ import annotations

from dataclasses import replace
import random
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from src.backends.base import GridPoint, Surface, build_direction_vectors, build_grid_points
from src.config import VanillaExecutionConfig
from src.data import Cifar10Dataset
from src.models.resnet20 import build_model as build_resnet20_model
from src.results import synchronize_device
from src.workloads import WORKLOADS


def _prepare_reference_model_and_data(
    config: VanillaExecutionConfig,
    device: torch.device,
):
    workload = config.workload
    if workload.task.name != "cifar10_resnet20_classification":
        raise ValueError(
            "original_algo only supports cifar10_resnet20_classification"
        )

    random.seed(workload.seed)
    torch.manual_seed(workload.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(workload.seed)
    if (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
    ):
        torch.mps.manual_seed(workload.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = build_resnet20_model(
        replace(
            WORKLOADS["cifar10_resnet20_classification"].spec,
            checkpoint_path=workload.task.checkpoint_path,
        )
    ).float()

    batch_size = (
        workload.data.gpu_batch_size
        if device.type != "cpu" and workload.data.gpu_batch_size is not None
        else workload.data.batch_size
    )
    data_loader = DataLoader(
        Cifar10Dataset(
            Path(WORKLOADS["cifar10_resnet20_classification"].spec.dataset_path),
            workload.data.subset_size,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workload.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = model.to(device)
    base_vector_cpu, direction_a_cpu, direction_b_cpu = build_direction_vectors(
        model,
        workload.seed,
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


def run_reference_surface(config: VanillaExecutionConfig) -> Surface:
    workload = config.workload
    device = torch.device(
        workload.runtime.device
        if workload.runtime.device != "auto"
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
    ) = _prepare_reference_model_and_data(config, device)

    points = build_grid_points(workload.grid)
    base_vector_device = base_vector_cpu.to(device)
    direction_a_device = direction_a_cpu.to(device)
    direction_b_device = direction_b_cpu.to(device)
    vector_to_parameters(base_vector_device, model.parameters())
    synchronize_device(device)

    return _evaluate_points_reference(
        model=model,
        data_loader=data_loader,
        device=device,
        chunk=points,
        base_vector_device=base_vector_device,
        direction_a_device=direction_a_device,
        direction_b_device=direction_b_device,
    )
