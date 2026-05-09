from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import Dataset

from src.system_schema import DatasetSpec, MLTaskSpec
from src.data import CaliforniaHousingDataset, Cifar10Dataset
from src.models.mlp_regressor import build_model as build_mlp_regressor_model
from src.models.resnet20 import build_model as build_resnet20_model

Batch = tuple[torch.Tensor, torch.Tensor]
DatasetBuilder = Callable[[MLTaskSpec, int], Dataset]
ModelBuilder = Callable[[MLTaskSpec], nn.Module]
LossComputer = Callable[[nn.Module, Batch, torch.device], tuple[torch.Tensor, int]]


@dataclass(frozen=True)
class WorkloadDefinition:
    spec: MLTaskSpec
    build_dataset: DatasetBuilder
    build_model: ModelBuilder
    compute_loss: LossComputer


def _build_cifar10_dataset(
    spec: MLTaskSpec,
    seed: int,
) -> Dataset:
    del seed
    return Cifar10Dataset(Path(spec.dataset.path), spec.dataset.sample_count)


def _build_california_housing_dataset(
    spec: MLTaskSpec,
    seed: int,
) -> Dataset:
    return CaliforniaHousingDataset(
        Path(spec.dataset.path),
        spec.dataset.sample_count,
        seed,
    )


def _compute_cross_entropy(
    model: nn.Module,
    batch: Batch,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    inputs, targets = batch
    inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    logits = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(logits, targets)
    return loss, int(targets.shape[0])


def _compute_mse(
    model: nn.Module,
    batch: Batch,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    inputs, targets = batch
    inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
    targets = targets.to(device, dtype=torch.float32, non_blocking=True)
    predictions = model(inputs).squeeze(-1)
    loss = torch.nn.MSELoss()(predictions, targets)
    return loss, int(targets.shape[0])


WORKLOADS: dict[str, WorkloadDefinition] = {
    "cifar10_resnet20_classification": WorkloadDefinition(
        MLTaskSpec(
            "cifar10_resnet20_classification",
            DatasetSpec(
                "cifar10",
                "assets/cifar-10-batches-py",
                (3, 32, 32),
                1024,
            ),
            "resnet20",
            "image_classification",
            "cross_entropy",
            "assets/cifar10-resnet20-0.pkl",
        ),
        _build_cifar10_dataset,
        build_resnet20_model,
        _compute_cross_entropy,
    ),
    "california_mlp_regression": WorkloadDefinition(
        MLTaskSpec(
            "california_mlp_regression",
            DatasetSpec(
                "california_housing",
                "assets",
                (8,),
                1024,
            ),
            "mlp_regressor",
            "tabular_regression",
            "mse",
            "assets/california-mlp-0.pkl",
        ),
        _build_california_housing_dataset,
        build_mlp_regressor_model,
        _compute_mse,
    ),
}
