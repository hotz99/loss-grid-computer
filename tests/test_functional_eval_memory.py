from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.functional_eval import memory
from src.functional_eval.memory import (
    SectionTimings,
    cuda_memory_snapshot,
    process_memory_snapshot,
    reset_cuda_peak_memory,
    time_block,
)


class FunctionalEvalMemoryTest(unittest.TestCase):
    def test_section_timings_defaults_to_zeroes(self) -> None:
        timings = SectionTimings()

        self.assertEqual(0.0, timings.perturbation_s)
        self.assertEqual(0.0, timings.binding_s)
        self.assertEqual(0.0, timings.batch_eval_s)
        self.assertEqual(0.0, timings.total_grid_s)

    def test_time_block_returns_non_negative_elapsed_time(self) -> None:
        sample = time_block()

        self.assertGreaterEqual(sample.elapsed_s, 0.0)
        self.assertGreaterEqual(sample.ended_at_s, sample.started_at_s)

    def test_cuda_memory_snapshot_gracefully_handles_unavailable_cuda(self) -> None:
        snapshot = cuda_memory_snapshot()

        if snapshot.available:
            self.assertIsInstance(snapshot.allocated_bytes, int)
            self.assertIsInstance(snapshot.reserved_bytes, int)
            self.assertIsInstance(snapshot.max_allocated_bytes, int)
            self.assertIsInstance(snapshot.max_reserved_bytes, int)
        else:
            self.assertIsInstance(snapshot.reason, str)
            self.assertGreater(len(snapshot.reason), 0)

    def test_reset_cuda_peak_memory_returns_boolean(self) -> None:
        self.assertIsInstance(reset_cuda_peak_memory(), bool)

    def test_cuda_memory_snapshot_handles_missing_cuda_api(self) -> None:
        fake_torch = SimpleNamespace()

        with patch.object(memory, "_import_torch", return_value=fake_torch):
            snapshot = cuda_memory_snapshot()

        self.assertFalse(snapshot.available)
        self.assertEqual("cuda is unavailable", snapshot.reason)

    def test_process_memory_snapshot_is_either_available_or_has_reason(self) -> None:
        snapshot = process_memory_snapshot()

        if snapshot.available:
            self.assertIsInstance(snapshot.rss_bytes, int)
            self.assertGreater(snapshot.rss_bytes, 0)
        else:
            self.assertIsInstance(snapshot.reason, str)
            self.assertGreater(len(snapshot.reason), 0)


if __name__ == "__main__":
    unittest.main()
