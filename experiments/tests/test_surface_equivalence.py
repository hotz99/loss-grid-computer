from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from experiments.candidates import GpuCandidate, run_standalone
from experiments.schemas import GridSpec, SurfaceGateConfig
from experiments.surface_gate import validate_surface
from experiments.workloads import task_for_workload


_GRID = GridSpec(resolution=2, scale=1.0)
_SEED = 42
_SAMPLE_COUNT = 64
_BATCH_SIZE = 32
_WORKLOAD = "mnist_mlp_classification"
_GATE = SurfaceGateConfig(rel_tol=1e-5, abs_tol=0.0)


def _assets_present() -> bool:
    return Path("assets/mnist/MNIST/raw/t10k-images-idx3-ubyte").exists()


def _run(candidate: GpuCandidate, device: torch.device):
    task = task_for_workload(_WORKLOAD, sample_count=_SAMPLE_COUNT)
    return run_standalone(candidate, task, _GRID, batch_size=_BATCH_SIZE, device=device, seed=_SEED)


def _check_surface(test, candidate_records, baseline_records, label: str) -> None:
    result = validate_surface(candidate_records, baseline_records, _GATE)
    test.assertEqual(
        0,
        result["mismatch_count"],
        f"{label}: {result['mismatch_count']} mismatches, "
        f"max_abs={result['max_abs_error']:.2e}",
    )


class CpuSurfaceEquivalenceTest(unittest.TestCase):
    def setUp(self) -> None:
        if not _assets_present():
            self.skipTest("MNIST assets not present")
        self._device = torch.device("cpu")

    def test_baseline_is_deterministic(self) -> None:
        r1 = _run(GpuCandidate.baseline(), self._device)
        r2 = _run(GpuCandidate.baseline(), self._device)
        _check_surface(self, r1.records, r2.records, "baseline repeat")

    def test_vmapped_matches_baseline(self) -> None:
        base = _run(GpuCandidate.baseline(), self._device)
        for k in (32, 64):
            with self.subTest(k=k):
                cand = _run(GpuCandidate.vmapped(k), self._device)
                _check_surface(self, cand.records, base.records, f"vmapped_k{k}")

    def test_compiled_matches_baseline(self) -> None:
        base = _run(GpuCandidate.baseline(), self._device)
        cand = _run(GpuCandidate.compiled(), self._device)
        self.assertIsNone(cand.error, f"compiled failed: {cand.error}")
        _check_surface(self, cand.records, base.records, "compiled")

    def test_compiled_vmapped_matches_baseline(self) -> None:
        base = _run(GpuCandidate.baseline(), self._device)
        for k in (32, 64):
            with self.subTest(k=k):
                cand = _run(GpuCandidate.compiled_vmapped(k), self._device)
                self.assertIsNone(cand.error, f"compiled_vmapped_k{k} failed: {cand.error}")
                _check_surface(self, cand.records, base.records, f"compiled_vmapped_k{k}")


class MpsSurfaceEquivalenceTest(unittest.TestCase):
    """Surface equivalence on the MPS backend.

    Guards the host->device transfer fix: candidates moved batches to MPS with
    ``non_blocking=True`` from pageable (non-pinned) memory, racing the compute
    kernel and producing garbage surfaces (worst under vmap, but the baseline
    backend shared the race). non_blocking is now gated on ``device.type ==
    "cuda"`` (the only path with pinned memory). A single run per candidate is
    enough: the regression reproduces on essentially every run.
    """

    def setUp(self) -> None:
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
        if not _assets_present():
            self.skipTest("MNIST assets not present")
        self._device = torch.device("mps")

    def test_baseline_is_deterministic(self) -> None:
        r1 = _run(GpuCandidate.baseline(), self._device)
        r2 = _run(GpuCandidate.baseline(), self._device)
        _check_surface(self, r1.records, r2.records, "baseline repeat")

    def test_vmapped_matches_baseline(self) -> None:
        base = _run(GpuCandidate.baseline(), self._device)
        for k in (32, 64):
            with self.subTest(k=k):
                cand = _run(GpuCandidate.vmapped(k), self._device)
                _check_surface(self, cand.records, base.records, f"vmapped_k{k}")

    def test_compiled_matches_baseline(self) -> None:
        base = _run(GpuCandidate.baseline(), self._device)
        cand = _run(GpuCandidate.compiled(), self._device)
        self.assertIsNone(cand.error, f"compiled failed: {cand.error}")
        _check_surface(self, cand.records, base.records, "compiled")

    def test_compiled_vmapped_matches_baseline(self) -> None:
        base = _run(GpuCandidate.baseline(), self._device)
        for k in (32, 64):
            with self.subTest(k=k):
                cand = _run(GpuCandidate.compiled_vmapped(k), self._device)
                self.assertIsNone(cand.error, f"compiled_vmapped_k{k} failed: {cand.error}")
                _check_surface(self, cand.records, base.records, f"compiled_vmapped_k{k}")


if __name__ == "__main__":
    unittest.main()
