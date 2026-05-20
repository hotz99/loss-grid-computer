from __future__ import annotations

import unittest

from experiments.functional_eval_experiments import (
    DEFAULT_FUNCTIONAL_EVAL_SAMPLE_COUNTS,
    DEFAULT_FUNCTIONAL_EVAL_WORKLOADS,
    FULL_TEST_SET_SCENARIO,
    PRD_VMAP_REPRODUCTION_SCENARIO,
    _best_valid_candidate_info,
    _filename_label,
    build_platform_scenarios,
    default_scenarios,
)
from src.functional_eval.experiment import (
    FunctionalEvalConfig,
    _candidate_specs,
    build_default_request,
)


class FunctionalEvalExperimentsTest(unittest.TestCase):
    def test_default_scenarios_cover_prd_and_scaling_runs(self) -> None:
        scenarios = default_scenarios()

        self.assertEqual(
            len(DEFAULT_FUNCTIONAL_EVAL_WORKLOADS)
            * len(DEFAULT_FUNCTIONAL_EVAL_SAMPLE_COUNTS),
            len(scenarios),
        )
        self.assertEqual(
            {
                (workload_name, sample_count)
                for workload_name in DEFAULT_FUNCTIONAL_EVAL_WORKLOADS
                for sample_count in DEFAULT_FUNCTIONAL_EVAL_SAMPLE_COUNTS
            },
            {
                (scenario.workload_name, scenario.sample_count)
                for scenario in scenarios
            },
        )
        for scenario in scenarios:
            self.assertIn(scenario.workload_name, scenario.name)
            self.assertEqual((), scenario.point_chunk_sizes)
            self.assertEqual(
                7 if scenario.sample_count == 1024 else 5,
                scenario.repeats,
            )
        self.assertEqual((1, 2, 4, 8, 16, 32, 64), PRD_VMAP_REPRODUCTION_SCENARIO.point_chunk_sizes)
        self.assertEqual((32, 64), FULL_TEST_SET_SCENARIO.point_chunk_sizes)
        self.assertEqual(0, FULL_TEST_SET_SCENARIO.sample_count)

    def test_filename_label_sanitizes_machine_discriminator(self) -> None:
        self.assertEqual("a100-80gb-run-1-", _filename_label("A100 80GB/run 1"))
        self.assertEqual("", _filename_label(None))

    def test_candidate_taxonomy_matches_exp_a_rationale(self) -> None:
        config = FunctionalEvalConfig(
            build_default_request(
                workload_name="mnist_mlp_classification",
                device="cpu",
                sample_count=2,
                batch_size=2,
                resolution=2,
            ),
            point_chunk_sizes=(1, 4),
        )

        specs = {spec.name: spec for spec in _candidate_specs(config)}

        sequential = specs["functional_sequential"]
        self.assertEqual(("functional_call",), sequential.components)
        self.assertEqual(("parameter_binding",), sequential.applies_to_sections)
        self.assertIn("does not vectorize forward/loss", sequential.hypothesis)

        vmapped = specs["vmapped_chunk_4"]
        self.assertEqual(
            ("functional_call", "vmap", "point_chunking"),
            vmapped.components,
        )
        self.assertEqual(
            (
                "perturbation_construction",
                "parameter_binding",
                "batch_forward_loss",
            ),
            vmapped.applies_to_sections,
        )
        self.assertIn(
            "perturbed variants of the original model",
            vmapped.hypothesis,
        )

    def test_best_valid_candidate_info_ignores_invalid_and_baseline_rows(self) -> None:
        # Multiple repeats per candidate are required for a 95% CI to be
        # computable; _best_valid_candidate_info now requires the lower CI
        # bound to exceed 1.0 (claim_status == "speedup") in addition to
        # surface validity.
        repeats = (0, 1, 2)
        summary = {
            "candidate_summary": [
                {
                    "candidate": "baseline_original",
                    "status_counts": {"ok": len(repeats)},
                },
                {
                    "candidate": "vmapped_chunk_32",
                    "status_counts": {"ok": len(repeats)},
                },
                {
                    "candidate": "vmapped_chunk_64",
                    "status_counts": {"ok": len(repeats)},
                },
                {
                    "candidate": "functional_sequential",
                    "status_counts": {"ok": len(repeats)},
                },
            ],
            "runs": [
                *[_run("baseline_original", r, 10.0) for r in repeats],
                *[_run("vmapped_chunk_32", r, 9.5, True) for r in repeats],
                *[_run("vmapped_chunk_64", r, 9.2, True) for r in repeats],
                # functional_sequential is faster on average but fails surface
                # validation, so it must be excluded regardless of speedup.
                *[_run("functional_sequential", r, 8.0, False, False) for r in repeats],
            ],
        }

        best = _best_valid_candidate_info(summary)

        self.assertIsNotNone(best)
        self.assertEqual("vmapped_chunk_64", best["candidate"])
        self.assertEqual("speedup", best["claim_status"])
        self.assertGreater(best["paired_speedup_mean"], 1.0)
        self.assertGreater(best["paired_speedup_ci_95_lo"], 1.0)

    def test_build_platform_scenarios_includes_requested_shape(self) -> None:
        scenarios = build_platform_scenarios(
            workload_names=("mnist_mlp_classification",),
            sample_counts=(4,),
            repeats=1,
            batch_size=2,
            resolution=2,
            point_chunk_sizes=(2,),
        )

        self.assertEqual(1, len(scenarios))
        scenario = scenarios[0]
        self.assertEqual("mnist_mlp_classification", scenario.workload_name)
        self.assertEqual(4, scenario.sample_count)
        self.assertEqual(1, scenario.repeats)
        self.assertEqual(2, scenario.batch_size)
        self.assertEqual(2, scenario.resolution)
        self.assertEqual((2,), scenario.point_chunk_sizes)


def _run(
    candidate: str,
    repeat: int,
    total_grid_s: float,
    validation: bool | None = None,
    within_budget: bool | None = None,
) -> dict[str, object]:
    return {
        "candidate": candidate,
        "repeat": repeat,
        "status": "ok",
        "total_grid_s": total_grid_s,
        "validation": (
            None
            if validation is None
            else {
                "allclose": validation,
                "max_abs_within_budget": (
                    validation if within_budget is None else within_budget
                ),
            }
        ),
    }


if __name__ == "__main__":
    unittest.main()
