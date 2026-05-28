from __future__ import annotations

import unittest

from experiments.exp_1_algorithm import plan
from experiments.schemas import Experiment1Config


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


if __name__ == "__main__":
    unittest.main()
