from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from experiments.schemas import DatasetSpec, MLTaskSpec


WORKLOADS: dict[str, MLTaskSpec] = {
    "cifar10_resnet20_classification": MLTaskSpec(
        name="cifar10_resnet20_classification",
        dataset=DatasetSpec("cifar10", "assets/cifar-10-batches-py", (3, 32, 32), 1024),
        model="resnet20",
        task="image_classification",
        loss="cross_entropy",
        checkpoint_path="assets/cifar10-resnet20-0.pkl",
    ),
    "cifar10_row_gru_classification": MLTaskSpec(
        name="cifar10_row_gru_classification",
        dataset=DatasetSpec("cifar10", "assets/cifar-10-batches-py", (3, 32, 32), 1024),
        model="row_gru",
        task="image_classification",
        loss="cross_entropy",
        checkpoint_path="assets/cifar10-row-gru-0.pkl",
    ),
    "california_mlp_regression": MLTaskSpec(
        name="california_mlp_regression",
        dataset=DatasetSpec("california_housing", "assets", (8,), 1024),
        model="mlp_regressor",
        task="tabular_regression",
        loss="mse",
        checkpoint_path="assets/california-mlp-0.pkl",
    ),
    "mnist_mlp_classification": MLTaskSpec(
        name="mnist_mlp_classification",
        dataset=DatasetSpec("mnist", "assets/mnist", (1, 28, 28), 1024),
        model="mlp",
        task="image_classification",
        loss="cross_entropy",
        checkpoint_path="assets/mnist-mlp-0.pkl",
    ),
}


def task_for_workload(name: str, sample_count: int | None = None) -> MLTaskSpec:
    task = WORKLOADS[name]
    if sample_count is None:
        return task
    return replace(
        task,
        dataset=replace(task.dataset, sample_count=sample_count),
    )


def workload_metadata(name: str, sample_count: int | None = None) -> dict[str, Any]:
    task = WORKLOADS.get(name)
    if task is None:
        return {"workload_name": name, "registered": False}
    task = task_for_workload(name, sample_count)
    payload = asdict(task)
    payload["workload_name"] = task.name
    payload["registered"] = True
    return payload
