from __future__ import annotations

import unittest

from experiments.exp_2_hybrid import plan
from experiments.schemas import Experiment2Config


class Experiment2ScaffoldTest(unittest.TestCase):
    def test_plan_alternates_pair_order(self) -> None:
        config = Experiment2Config(
            workload_names=("mnist_mlp_classification",),
            repeats=2,
        )

        trials = plan(config)

        self.assertEqual(4, len(trials))
        self.assertEqual(("vanilla", "hybrid"), trials[0].trial_order)
        self.assertEqual(("hybrid", "vanilla"), trials[2].trial_order)


if __name__ == "__main__":
    unittest.main()
