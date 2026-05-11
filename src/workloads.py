from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _asset(relative: str) -> str:
    return str(_REPO_ROOT / relative)

import torch
from torch import nn
from torch.utils.data import Dataset

from src.schemas import DatasetSpec, MLTaskSpec
from src.data import CaliforniaHousingDataset, Cifar10Dataset, MnistDataset
from src.models.mnist_mlp import build_model as build_mnist_mlp_model
from src.models.mlp_regressor import build_model as build_mlp_regressor_model
from src.models.resnet20 import build_model as build_resnet20_model
from src.models.row_gru import build_model as build_row_gru_model

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


def _build_mnist_dataset(
    spec: MLTaskSpec,
    seed: int,
) -> Dataset:
    del seed
    return MnistDataset(Path(spec.dataset.path), spec.dataset.sample_count)


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
                _asset("assets/cifar-10-batches-py"),
                (3, 32, 32),
                1024,
            ),
            "resnet20",
            "image_classification",
            "cross_entropy",
            _asset("assets/cifar10-resnet20-0.pkl"),
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
                _asset("assets"),
                (8,),
                1024,
            ),
            "mlp_regressor",
            "tabular_regression",
            "mse",
            _asset("assets/california-mlp-0.pkl"),
        ),
        _build_california_housing_dataset,
        build_mlp_regressor_model,
        _compute_mse,
    ),
    "cifar10_row_gru_classification": WorkloadDefinition(
        MLTaskSpec(
            "cifar10_row_gru_classification",
            DatasetSpec(
                "cifar10",
                _asset("assets/cifar-10-batches-py"),
                (3, 32, 32),
                1024,
            ),
            "row_gru",
            "image_classification",
            "cross_entropy",
            _asset("assets/cifar10-row-gru-0.pkl"),
        ),
        _build_cifar10_dataset,
        build_row_gru_model,
        _compute_cross_entropy,
    ),
    "mnist_mlp_classification": WorkloadDefinition(
        MLTaskSpec(
            "mnist_mlp_classification",
            DatasetSpec(
                "mnist",
                _asset("assets/mnist"),
                (1, 28, 28),
                1024,
            ),
            "mnist_mlp",
            "image_classification",
            "cross_entropy",
            _asset("assets/mnist-mlp-0.pkl"),
        ),
        _build_mnist_dataset,
        build_mnist_mlp_model,
        _compute_cross_entropy,
    ),
}
