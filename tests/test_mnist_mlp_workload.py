from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.models import build_model
from src.system_schema import DatasetSpec, MLTaskSpec
from src.workloads import WORKLOADS


MNIST_MLP_WORKLOAD = "mnist_mlp_classification"
MNIST_MLP_CHECKPOINT = Path("assets/mnist-mlp-0.pkl")


class MnistMLPWorkloadTest(unittest.TestCase):
    def test_model_builds_without_checkpoint(self) -> None:
        model = build_model(_task_spec(checkpoint_path=None))

        self.assertIsInstance(model, torch.nn.Module)

    def test_forward_shape_maps_mnist_batch_to_ten_logits(self) -> None:
        model = build_model(_task_spec(checkpoint_path=None))
        inputs = torch.randn(2, 1, 28, 28)

        with torch.no_grad():
            logits = model(inputs)

        self.assertEqual((2, 10), tuple(logits.shape))

    def test_workload_is_registered(self) -> None:
        definition = WORKLOADS[MNIST_MLP_WORKLOAD]

        self.assertEqual(MNIST_MLP_WORKLOAD, definition.spec.name)
        self.assertEqual("mnist", definition.spec.dataset.name)
        self.assertEqual((1, 28, 28), definition.spec.dataset.input_shape)
        self.assertEqual("mnist_mlp", definition.spec.model)
        self.assertEqual("image_classification", definition.spec.task)
        self.assertEqual("cross_entropy", definition.spec.loss)
        self.assertEqual(str(MNIST_MLP_CHECKPOINT), definition.spec.checkpoint_path)


def _task_spec(*, checkpoint_path: str | None) -> MLTaskSpec:
    return MLTaskSpec(
        MNIST_MLP_WORKLOAD,
        DatasetSpec(
            "mnist",
            "assets/mnist",
            (1, 28, 28),
            1024,
        ),
        "mnist_mlp",
        "image_classification",
        "cross_entropy",
        checkpoint_path,
    )


if __name__ == "__main__":
    unittest.main()
