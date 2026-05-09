from __future__ import annotations

import math
import unittest
from pathlib import Path

from src.backends import vanilla
from src.system_schema import (
    DatasetSpec,
    GridSpec,
    MLTaskSpec,
    SchedulerRequest,
    VanillaMode,
)
from src.original_algo import run_reference_surface


class BackendEquivalenceTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        if not Path("assets/cifar10-resnet20-0.pkl").exists():
            self.skipTest("ResNet20 checkpoint is not available")
        if not Path("assets/cifar-10-batches-py/test_batch").exists():
            self.skipTest("CIFAR-10 test batch is not available")

    def test_vanilla_repeated_runs_match_on_4x4_grid(self) -> None:
        first = _run_records(_config())
        second = _run_records(_config())
        self.assertEqual([], _mismatches(first, second, "first", "second")[:8])

    def test_vanilla_matches_original_algo_on_4x4_grid(self) -> None:
        config = _config()
        current = _run_records(config)
        reference = _sorted_records(
            run_reference_surface(
                config,
                seed=1337,
            )
        )
        self.assertEqual([], _mismatches(current, reference, "current", "reference")[:8])


def _config() -> SchedulerRequest:
    return SchedulerRequest(
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
        GridSpec(4, 1.0),
        VanillaMode(32),
        "cpu",
    )


def _same_loss(lhs: float, rhs: float) -> bool:
    if math.isnan(lhs) or math.isnan(rhs):
        return math.isnan(lhs) and math.isnan(rhs)
    if math.isinf(lhs) or math.isinf(rhs):
        return math.isinf(lhs) and math.isinf(rhs) and (lhs > 0) == (rhs > 0)
    return math.isclose(lhs, rhs, rel_tol=1e-5, abs_tol=1e-6)


def _sorted_records(records: list[tuple[int, int, float]]):
    return sorted(records, key=lambda record: (record[0], record[1]))


def _run_records(config: SchedulerRequest) -> list[tuple[int, int, float]]:
    return _sorted_records(
        vanilla.run(
            config,
            seed=1337,
        ).records or []
    )


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

if __name__ == "__main__":
    unittest.main()
