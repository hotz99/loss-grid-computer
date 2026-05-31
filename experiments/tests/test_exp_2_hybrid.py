from __future__ import annotations

import unittest

from experiments.exp_2_hybrid import (
    _b_claim_status,
    _slowdown_ladder,
    _threshold_summary,
    plan,
)
from experiments.schemas import Experiment2Config


def _rung(slowdown: int, low: float | None, high: float | None,
          achieved: float | None = None) -> dict:
    return {
        "slowdown_factor": slowdown,
        "speedup_ci_low": low,
        "speedup_ci_high": high,
        "achieved_ratio": achieved,
    }


class Experiment2ScaffoldTest(unittest.TestCase):
    def test_plan_sweeps_full_ladder_per_workload(self) -> None:
        config = Experiment2Config(
            workload_names=("mnist_mlp_classification",),
            repeats=2,
            slowdown_ceiling=4,
        )

        trials = plan(config)

        # 3 rungs {1, 2, 4} x R=2 x 2 candidates per repeat.
        self.assertEqual(12, len(trials))
        self.assertEqual(({"slowdown": 1}), trials[0].control)
        self.assertEqual(("vanilla", "hybrid"), trials[0].trial_order)
        self.assertEqual(("hybrid", "vanilla"), trials[2].trial_order)
        slowdowns = sorted({t.control["slowdown"] for t in trials})
        self.assertEqual([1, 2, 4], slowdowns)

    def test_ladder_is_base_two_and_starts_at_one(self) -> None:
        self.assertEqual(
            (1, 2, 4, 8, 16),
            _slowdown_ladder(Experiment2Config(slowdown_ceiling=16)),
        )
        # Ceiling caps the top rung; the ladder always starts at slow=1.
        self.assertEqual((1, 2, 4), _slowdown_ladder(Experiment2Config(slowdown_ceiling=5)))
        self.assertEqual((1,), _slowdown_ladder(Experiment2Config(slowdown_ceiling=1)))


class ClaimStatusTest(unittest.TestCase):
    def test_claim_status_maps_base_verdicts(self) -> None:
        self.assertEqual("hybrid_wins", _b_claim_status("speedup"))
        self.assertEqual("hybrid_regresses", _b_claim_status("regression"))
        self.assertEqual("inconclusive", _b_claim_status("inconclusive"))
        self.assertEqual("inconclusive", _b_claim_status("insufficient_data"))
        self.assertEqual("invalid_surface", _b_claim_status("invalid_surface"))

    def test_predictor_invalid_status_is_removed(self) -> None:
        # r_native plays no verdict role, so no rung can be predictor_invalid.
        for base in ("speedup", "regression", "inconclusive",
                     "insufficient_data", "invalid_surface"):
            self.assertNotEqual("predictor_invalid", _b_claim_status(base))


class ThresholdSummaryTest(unittest.TestCase):
    def test_first_crossing_within_range(self) -> None:
        ladder = [
            _rung(1, 0.8, 0.95),
            _rung(2, 0.9, 1.05),
            _rung(4, 1.2, 1.5, achieved=0.7),
            _rung(8, 1.4, 1.9),
        ]
        summary = _threshold_summary(ladder)
        self.assertEqual(4, summary["threshold_slowdown"])
        self.assertEqual("crosses_within_range", summary["threshold_status"])
        self.assertEqual([2, 4], summary["threshold_bracket"])
        self.assertEqual(0.7, summary["achieved_ratio_at_threshold"])

    def test_wins_at_native(self) -> None:
        ladder = [
            _rung(1, 1.1, 1.3, achieved=0.9),
            _rung(2, 1.5, 1.8),
        ]
        summary = _threshold_summary(ladder)
        self.assertEqual(1, summary["threshold_slowdown"])
        self.assertEqual("wins_at_native", summary["threshold_status"])
        self.assertEqual([None, 1], summary["threshold_bracket"])
        self.assertEqual(0.9, summary["achieved_ratio_at_threshold"])

    def test_above_explored_range(self) -> None:
        ladder = [
            _rung(1, 0.7, 0.9),
            _rung(2, 0.85, 1.02),
            _rung(4, 0.9, 1.1),
        ]
        summary = _threshold_summary(ladder)
        self.assertIsNone(summary["threshold_slowdown"])
        self.assertEqual("above_explored_range", summary["threshold_status"])
        self.assertIsNone(summary["threshold_bracket"])
        self.assertIsNone(summary["achieved_ratio_at_threshold"])

    def test_non_monotone_when_higher_rung_regresses(self) -> None:
        ladder = [
            _rung(1, 0.8, 0.95),
            _rung(2, 1.2, 1.4, achieved=0.6),
            _rung(4, 0.5, 0.8),  # regresses after a win
        ]
        summary = _threshold_summary(ladder)
        self.assertEqual(2, summary["threshold_slowdown"])
        self.assertEqual("non_monotone", summary["threshold_status"])
        self.assertIsNone(summary["threshold_bracket"])
        self.assertEqual(0.6, summary["achieved_ratio_at_threshold"])

    def test_inconclusive_rung_does_not_bracket(self) -> None:
        # A rung whose CI contains 1.0 neither confirms nor brackets; the
        # threshold is the first rung that clears it.
        ladder = [
            _rung(1, 0.9, 1.1),
            _rung(2, 1.05, 1.3),
        ]
        summary = _threshold_summary(ladder)
        self.assertEqual(2, summary["threshold_slowdown"])
        self.assertEqual("crosses_within_range", summary["threshold_status"])
        self.assertEqual([1, 2], summary["threshold_bracket"])


if __name__ == "__main__":
    unittest.main()
