from __future__ import annotations

import argparse
import math
import os
import sys
import unittest
from pathlib import Path
from typing import Literal

import torch

from src.backends import vanilla
from src.config import (
    DataConfig,
    ExperimentConfig,
    GridConfig,
    ModelConfig,
    ResourcesConfig,
    RuntimeConfig,
)


DeviceName = Literal["cpu", "mps", "cuda"]
SELECTED_DEVICE: DeviceName = os.environ.get("LOSS_GRID_TEST_DEVICE", "cpu")  # type: ignore[assignment]


class BackendEquivalenceTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        if not Path("assets/cifar10-resnet20-0.pkl").exists():
            self.skipTest("ResNet20 checkpoint is not available")
        if not Path("assets/cifar-10-batches-py/test_batch").exists():
            self.skipTest("CIFAR-10 test batch is not available")
        if SELECTED_DEVICE == "mps" and not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available")
        if SELECTED_DEVICE == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA backend is not available")

    def test_vanilla_repeated_runs_match_on_4x4_grid(self) -> None:
        first = _run_records(_config(device=SELECTED_DEVICE))
        second = _run_records(_config(device=SELECTED_DEVICE))
        self.assertEqual([], _mismatches(first, second, "first", "second")[:8])


def _config(device: DeviceName) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="backend-equivalence",
        seed=1337,
        backend="vanilla",
        model=ModelConfig(checkpoint_path="assets/cifar10-resnet20-0.pkl"),
        data=DataConfig(
            subset_size=128,
            batch_size=32,
            cpu_batch_size=32,
            num_workers=0,
        ),
        grid=GridConfig(resolution=4, scale=1.0),
        runtime=RuntimeConfig(
            device=device,
            num_batches=None,
            preload=False,
            gpu_slowdown_factor=1.0,
        ),
        resources=ResourcesConfig(cpu_workers=0),
    )


def _same_loss(lhs: float, rhs: float) -> bool:
    if math.isnan(lhs) or math.isnan(rhs):
        return math.isnan(lhs) and math.isnan(rhs)
    if math.isinf(lhs) or math.isinf(rhs):
        return math.isinf(lhs) and math.isinf(rhs) and (lhs > 0) == (rhs > 0)
    return math.isclose(lhs, rhs, rel_tol=1e-5, abs_tol=1e-6)


def _sorted_records(records: list[tuple[int, int, float]]):
    return sorted(records, key=lambda record: (record[0], record[1]))


def _run_records(config: ExperimentConfig) -> list[tuple[int, int, float]]:
    return _sorted_records(vanilla.run(config).records or [])


def _mismatches(
    lhs_records: list[tuple[int, int, float]],
    rhs_records: list[tuple[int, int, float]],
    lhs_label: str,
    rhs_label: str,
):
    assert len(lhs_records) == len(rhs_records)
    mismatches = []
    for lhs, rhs in zip(lhs_records, rhs_records):
        assert lhs[:2] == rhs[:2]
        row, col, lhs_loss = lhs
        _, _, rhs_loss = rhs
        if _same_loss(lhs_loss, rhs_loss):
            continue
        mismatches.append(
            {
                "row": row,
                "col": col,
                lhs_label: lhs_loss,
                rhs_label: rhs_loss,
            }
        )
    return mismatches


def _parse_device(argv: list[str]) -> tuple[DeviceName, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args, remaining = parser.parse_known_args(argv)
    return args.device, remaining


if __name__ == "__main__":
    SELECTED_DEVICE, remaining_argv = _parse_device(sys.argv[1:])
    unittest.main(argv=[sys.argv[0], *remaining_argv])
