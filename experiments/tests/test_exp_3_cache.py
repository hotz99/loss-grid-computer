from __future__ import annotations

import unittest

from experiments.calibration import CalibratedCell
from experiments.candidates import GpuCandidate
from experiments.exp_3_cache import _measure_compile_cost
from experiments.sessions import break_even_n


class BreakEvenOneTimeCostTest(unittest.TestCase):
    def test_compile_is_added_to_the_amortized_numerator(self) -> None:
        # T_v=10, T_p=8 -> margin 2/grid. calibration 4s alone breaks even at 2;
        # folding a 6s compile lifts the one-time cost to 10s -> 5 grids.
        self.assertEqual(2, break_even_n(10.0, 8.0, 4.0))
        self.assertEqual(5, break_even_n(10.0, 8.0, 4.0 + 6.0))

    def test_absent_when_warm_session_is_not_faster(self) -> None:
        self.assertIsNone(break_even_n(8.0, 8.0, 10.0))
        self.assertIsNone(break_even_n(7.0, 8.0, 10.0))


class MeasureCompileCostTest(unittest.TestCase):
    def test_non_compiling_roles_cost_zero_without_running(self) -> None:
        # baseline/vmapped never compile: the helper must short-circuit to 0.0
        # so the one-time setup cost stays uniform across workloads, and it must
        # not invoke run_standalone (no task/device needed here).
        for candidate in (GpuCandidate.baseline(), GpuCandidate.vmapped(32)):
            cost = _measure_compile_cost(
                candidate, task=None, grid=None,
                gpu_batch_size=32, device=None, seed=0, gpu_slowdown_factor=1.0,
            )
            self.assertEqual(0.0, cost)


class OneTimeSetupCellTest(unittest.TestCase):
    def test_calibration_cell_carries_calibration_s(self) -> None:
        # Guards the field the session reads when folding compile into setup.
        cell = CalibratedCell(
            selected_policy="gpu_only", gpu_batch_size=32, cpu_batch_size=None,
            cpu_workers=None, calibration_s=4.0, baseline_total_s=10.0,
            selected_total_s=None,
        )
        self.assertEqual(4.0, cell.calibration_s)


if __name__ == "__main__":
    unittest.main()
