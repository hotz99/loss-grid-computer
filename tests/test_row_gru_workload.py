from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.backends import vanilla
from src.models import build_model
from src.system_schema import (
    DatasetSpec,
    GridSpec,
    MLTaskSpec,
    SchedulerRequest,
    VanillaMode,
)
from src.workloads import WORKLOADS


ROW_GRU_WORKLOAD = "cifar10_row_gru_classification"
ROW_GRU_CHECKPOINT = Path("assets/cifar10-row-gru-0.pkl")
CIFAR_TEST_BATCH = Path("assets/cifar-10-batches-py/test_batch")


class RowGRUWorkloadTest(unittest.TestCase):
    def test_model_builds_without_checkpoint(self) -> None:
        model = build_model(_task_spec(checkpoint_path=None))

        self.assertIsInstance(model, torch.nn.Module)

    def test_forward_shape_maps_cifar_batch_to_ten_logits(self) -> None:
        model = build_model(_task_spec(checkpoint_path=None))
        inputs = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            logits = model(inputs)

        self.assertEqual((2, 10), tuple(logits.shape))

    def test_workload_is_registered(self) -> None:
        definition = WORKLOADS[ROW_GRU_WORKLOAD]

        self.assertEqual(ROW_GRU_WORKLOAD, definition.spec.name)
        self.assertEqual("cifar10", definition.spec.dataset.name)
        self.assertEqual((3, 32, 32), definition.spec.dataset.input_shape)
        self.assertEqual("row_gru", definition.spec.model)
        self.assertEqual("image_classification", definition.spec.task)
        self.assertEqual("cross_entropy", definition.spec.loss)
        self.assertEqual(str(ROW_GRU_CHECKPOINT), definition.spec.checkpoint_path)

    def test_vanilla_run_smoke_on_tiny_grid(self) -> None:
        if not ROW_GRU_CHECKPOINT.exists():
            self.skipTest("Row-GRU checkpoint is not available")
        if not CIFAR_TEST_BATCH.exists():
            self.skipTest("CIFAR-10 test batch is not available")

        result = vanilla.run(
            SchedulerRequest(
                _task_spec(
                    checkpoint_path=str(ROW_GRU_CHECKPOINT),
                    sample_count=2,
                ),
                GridSpec(resolution=1, scale=0.0),
                VanillaMode(gpu_batch_size=2),
                "cpu",
            ),
            seed=1337,
        )

        self.assertEqual([(0, 0)], [(row, col) for row, col, _ in result.records or []])
        for _, _, loss in result.records or []:
            self.assertTrue(torch.isfinite(torch.tensor(loss)))


def _task_spec(
    *,
    checkpoint_path: str | None,
    sample_count: int = 1024,
) -> MLTaskSpec:
    return MLTaskSpec(
        ROW_GRU_WORKLOAD,
        DatasetSpec(
            "cifar10",
            "assets/cifar-10-batches-py",
            (3, 32, 32),
            sample_count,
        ),
        "row_gru",
        "image_classification",
        "cross_entropy",
        checkpoint_path,
    )


if __name__ == "__main__":
    unittest.main()
