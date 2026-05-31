from __future__ import annotations

import unittest

from experiments.calibration import CalibratedCell
from experiments.candidates import GpuCandidate
from experiments.exp_3_cache import _measure_compile_cost, _select_from_experiment_2
from experiments.schemas import Experiment2Config, Experiment2Result
from experiments.sessions import break_even_n


def _exp2(workloads: dict) -> Experiment2Result:
    return Experiment2Result(
        status="completed",
        schema_version="experiment-2-hybrid-v1",
        config=Experiment2Config(),
        result={},
        record={"workloads": workloads},
    )


def _workload(status_threshold: str, slowdown, ladder, achieved=None) -> dict:
    return {
        "status": "completed",
        "threshold_status": status_threshold,
        "threshold_slowdown": slowdown,
        "achieved_ratio_at_threshold": achieved,
        "ladder": ladder,
    }


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


class SelectFromExperiment2Test(unittest.TestCase):
    def test_none_when_no_threshold_reached(self) -> None:
        exp2 = _exp2(
            {
                "w_a": _workload("above_explored_range", None, [
                    {"slowdown_factor": 1, "speedup_ci_low": 0.8},
                ]),
            }
        )
        self.assertIsNone(_select_from_experiment_2(exp2))

    def test_selects_reached_threshold_at_its_operating_point(self) -> None:
        exp2 = _exp2(
            {
                "w_a": _workload("crosses_within_range", 4, [
                    {"slowdown_factor": 1, "speedup_ci_low": 0.8},
                    {"slowdown_factor": 4, "speedup_ci_low": 1.3},
                ], achieved=0.7),
            }
        )
        sel = _select_from_experiment_2(exp2)
        self.assertEqual("w_a", sel["workload_name"])
        self.assertEqual(4.0, sel["slowdown_factor"])
        self.assertEqual("slowed", sel["operating_point"])
        self.assertEqual(0.7, sel["achieved_ratio_at_threshold"])

    def test_prefers_lowest_threshold_then_native_operating_point(self) -> None:
        exp2 = _exp2(
            {
                "slowed": _workload("crosses_within_range", 4, [
                    {"slowdown_factor": 4, "speedup_ci_low": 2.0},
                ]),
                "native": _workload("wins_at_native", 1, [
                    {"slowdown_factor": 1, "speedup_ci_low": 1.1},
                ]),
            }
        )
        sel = _select_from_experiment_2(exp2)
        self.assertEqual("native", sel["workload_name"])
        self.assertEqual(1.0, sel["slowdown_factor"])
        self.assertEqual("native", sel["operating_point"])

    def test_non_monotone_threshold_is_still_selectable(self) -> None:
        exp2 = _exp2(
            {
                "w_a": _workload("non_monotone", 2, [
                    {"slowdown_factor": 2, "speedup_ci_low": 1.2},
                ]),
            }
        )
        sel = _select_from_experiment_2(exp2)
        self.assertEqual("w_a", sel["workload_name"])
        self.assertEqual(2.0, sel["slowdown_factor"])


if __name__ == "__main__":
    unittest.main()
