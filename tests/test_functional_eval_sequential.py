from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset

from src.backends.base import GridPoint
from src.functional_eval.sequential import (
    evaluate_points_functional_with_definition,
    run_functional_sequential_surface,
)
from src.functional_eval.validation import compare_surfaces
from src.original_algo import run_reference_surface
from src.system_schema import DatasetSpec, GridSpec, MLTaskSpec, SchedulerRequest, VanillaMode
from src.workloads import WorkloadDefinition


class FunctionalSequentialEvalTest(unittest.TestCase):
    def test_matches_mutating_reference_on_synthetic_workload(self) -> None:
        device = torch.device("cpu")
        model = _TinyClassifier().to(device).eval()
        inputs = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        data_loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
        base_vector = parameters_to_vector(model.parameters()).detach()
        direction_a = torch.linspace(
            -0.2,
            0.3,
            base_vector.numel(),
            dtype=base_vector.dtype,
        )
        direction_b = torch.linspace(
            0.1,
            -0.25,
            base_vector.numel(),
            dtype=base_vector.dtype,
        )
        points = (
            GridPoint(0, 0, 0, -0.5, -0.5),
            GridPoint(1, 0, 1, -0.5, 0.5),
            GridPoint(2, 1, 0, 0.5, -0.5),
            GridPoint(3, 1, 1, 0.5, 0.5),
        )

        functional = evaluate_points_functional_with_definition(
            definition=_tiny_workload_definition(),
            model=model,
            data_loader=data_loader,
            device=device,
            chunk=points,
            base_vector_device=base_vector,
            direction_a_device=direction_a,
            direction_b_device=direction_b,
        )
        mutating = _evaluate_points_mutating(
            model=_TinyClassifier().to(device).eval(),
            data_loader=data_loader,
            device=device,
            points=points,
            base_vector=base_vector,
            direction_a=direction_a,
            direction_b=direction_b,
        )

        comparison = compare_surfaces(functional.records, mutating)
        self.assertTrue(comparison.allclose, comparison)
        self.assertEqual(4, len(functional.records))
        self.assertGreaterEqual(functional.timings.perturbation_s, 0.0)
        self.assertGreaterEqual(functional.timings.binding_s, 0.0)
        self.assertGreaterEqual(functional.timings.batch_eval_s, 0.0)
        self.assertGreaterEqual(functional.timings.total_grid_s, 0.0)

    def test_matches_original_algo_on_tiny_cifar_grid_when_assets_exist(self) -> None:
        if not Path("assets/cifar10-resnet20-0.pkl").exists():
            self.skipTest("ResNet20 checkpoint is not available")
        if not Path("assets/cifar-10-batches-py/test_batch").exists():
            self.skipTest("CIFAR-10 test batch is not available")

        config = _tiny_cifar_config()
        functional = run_functional_sequential_surface(config, seed=1337)
        reference = run_reference_surface(config, seed=1337)

        comparison = compare_surfaces(functional.records, reference)
        self.assertTrue(comparison.allclose, comparison)
        self.assertEqual(4, comparison.point_count)


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _tiny_workload_definition() -> WorkloadDefinition:
    return WorkloadDefinition(
        spec=MLTaskSpec(
            "tiny_classification",
            DatasetSpec("tensor", "", (3,), 4),
            "tiny",
            "classification",
            "cross_entropy",
            None,
        ),
        build_dataset=lambda _spec, _seed: TensorDataset(),
        build_model=lambda _spec: _TinyClassifier(),
        compute_loss=_compute_tiny_cross_entropy,
    )


def _compute_tiny_cross_entropy(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    inputs, targets = batch
    inputs = inputs.to(device, dtype=torch.float32)
    targets = targets.to(device)
    return torch.nn.CrossEntropyLoss()(model(inputs), targets), int(targets.shape[0])


def _evaluate_points_mutating(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    points: tuple[GridPoint, ...],
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
) -> list[tuple[int, int, float]]:
    records: list[tuple[int, int, float]] = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for point in points:
        perturbed = base_vector + point.alpha * direction_a + point.beta * direction_b
        vector_to_parameters(perturbed, model.parameters())
        total_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                logits = model(inputs.to(device))
                loss = loss_fn(logits, targets.to(device))
                batch_size = int(targets.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size
        records.append((point.row, point.col, total_loss / max(1, total_examples)))
    return records


def _tiny_cifar_config() -> SchedulerRequest:
    task = MLTaskSpec(
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
    )
    return SchedulerRequest(
        replace(task, dataset=replace(task.dataset, sample_count=4)),
        GridSpec(2, 0.1),
        VanillaMode(2),
        "cpu",
    )


if __name__ == "__main__":
    unittest.main()
