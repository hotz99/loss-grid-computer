from __future__ import annotations

import unittest

from experiments.exp_1_algorithm import _compile_amortization, plan
from experiments.schemas import Experiment1Config, GridSpec
from experiments.stats import break_even_points, cold_inclusive_speedups


class Experiment1ScaffoldTest(unittest.TestCase):
    def test_plan_rotates_baseline_and_vmapped_roles(self) -> None:
        config = Experiment1Config(
            workload_names=("mnist_mlp_classification",),
            repeats=2,
            point_chunk_sizes=(32, 64),
        )

        trials = plan(config)

        self.assertEqual(6, len(trials))
        self.assertEqual(("baseline", "vmapped"), trials[0].trial_order)
        self.assertEqual(("vmapped", "baseline"), trials[3].trial_order)
        self.assertEqual(
            {"baseline", "vmapped_k32", "vmapped_k64"},
            {trial.candidate for trial in trials},
        )


class CompileAmortizationStatsTest(unittest.TestCase):
    def test_break_even_and_cold_inclusive(self) -> None:
        # N=10, baseline 1.0/pt, warm candidate 0.2/pt (5x), compile 4.0s.
        # break-even n* = C*N/(T_base - T_cand) = 4*10/(10-2) = 5 points.
        base = {0: 10.0, 1: 10.0}
        cand = {0: 2.0, 1: 2.0}
        comp = {0: 4.0, 1: 4.0}
        self.assertEqual([5.0, 5.0], break_even_points(base, cand, comp, 10))
        cold = cold_inclusive_speedups(base, cand, comp)
        self.assertEqual(2, len(cold))
        self.assertAlmostEqual(10.0 / 6.0, cold[0])

    def test_break_even_drops_repeats_where_warm_is_slower(self) -> None:
        # Warm candidate slower than baseline: compile never amortizes.
        self.assertEqual([], break_even_points({0: 1.0}, {0: 2.0}, {0: 4.0}, 10))


class CompileAmortizationAggregateTest(unittest.TestCase):
    def test_amortization_block_populated_for_compiling_candidate(self) -> None:
        config = Experiment1Config(repeats=2, grid=GridSpec(8, 1.0))  # 64 grid points
        workload, candidate = "mnist_mlp_classification", "compiled_vmapped_k64"
        baseline_times = {0: 64.0, 1: 64.0}
        candidate_times = {0: 16.0, 1: 16.0}  # warm 4x
        raw_diagnostics = {
            (workload, candidate, 0): {"compile_cold_start_s": 6.0, "recompile_count": 0},
            (workload, candidate, 1): {"compile_cold_start_s": 6.0, "recompile_count": 0},
        }

        amort = _compile_amortization(
            workload, candidate, config, raw_diagnostics,
            baseline_times, candidate_times, grid_points=64,
        )

        assert amort is not None
        self.assertEqual([6.0, 6.0], amort["compile_cold_start_s"])
        self.assertEqual(0, amort["recompile_count_max"])
        # break-even = 6*64/(64-16) = 8 points, well within the 64-point grid.
        self.assertAlmostEqual(8.0, amort["break_even_geomean"])
        self.assertTrue(amort["amortizes_within_grid"])
        self.assertAlmostEqual(64.0 / 22.0, amort["cold_inclusive_geomean"])

    def test_amortization_absent_without_compile_diagnostics(self) -> None:
        config = Experiment1Config(repeats=2, grid=GridSpec(8, 1.0))
        amort = _compile_amortization(
            "mnist_mlp_classification", "vmapped_k64", config, raw_diagnostics={},
            baseline_times={0: 10.0}, candidate_times={0: 2.0}, grid_points=64,
        )
        self.assertIsNone(amort)


if __name__ == "__main__":
    unittest.main()
