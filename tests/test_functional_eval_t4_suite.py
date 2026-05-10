from __future__ import annotations

import unittest

from src.functional_eval.platform_suite import (
    FULL_TEST_SET_SCENARIO,
    PRD_VMAP_REPRODUCTION_SCENARIO,
)
from src.functional_eval.t4_suite import (
    _acceptance,
    _best_valid_candidate,
    _filename_label,
    default_scenarios,
)


class FunctionalEvalPlatformSuiteTest(unittest.TestCase):
    def test_default_scenarios_cover_prd_and_scaling_runs(self) -> None:
        scenarios = default_scenarios()

        self.assertEqual(
            ["functional_seq_1024_stability", "functional_seq_2k_stability"],
            [scenario.name for scenario in scenarios],
        )
        self.assertEqual(7, scenarios[0].repeats)
        self.assertEqual(5, scenarios[1].repeats)
        self.assertEqual((), scenarios[0].point_chunk_sizes)
        self.assertEqual((), scenarios[1].point_chunk_sizes)
        self.assertEqual((1, 2, 4, 8, 16, 32, 64), PRD_VMAP_REPRODUCTION_SCENARIO.point_chunk_sizes)
        self.assertEqual((32, 64), FULL_TEST_SET_SCENARIO.point_chunk_sizes)
        self.assertEqual(0, FULL_TEST_SET_SCENARIO.sample_count)

    def test_filename_label_sanitizes_machine_discriminator(self) -> None:
        self.assertEqual("a100-80gb-run-1-", _filename_label("A100 80GB/run 1"))
        self.assertEqual("", _filename_label(None))

    def test_best_valid_candidate_ignores_invalid_and_baseline_rows(self) -> None:
        summary = {
            "candidate_summary": [
                {
                    "candidate": "baseline_original",
                    "taxonomy": {
                        "components": ["original_in_place_mutation"],
                    },
                    "status_counts": {"ok": 3},
                    "all_validations_passed": None,
                    "mean_speedup_vs_baseline": None,
                },
                {
                    "candidate": "vmapped_chunk_32",
                    "taxonomy": {
                        "components": ["functional_call", "vmap", "point_chunking"],
                    },
                    "status_counts": {"ok": 3},
                    "all_validations_passed": True,
                    "mean_speedup_vs_baseline": 1.06,
                },
                {
                    "candidate": "vmapped_chunk_64",
                    "taxonomy": {
                        "components": ["functional_call", "vmap", "point_chunking"],
                    },
                    "status_counts": {"ok": 3},
                    "all_validations_passed": True,
                    "mean_speedup_vs_baseline": 1.08,
                },
                {
                    "candidate": "vmapped_chunk_128",
                    "taxonomy": {
                        "components": ["functional_call", "vmap", "point_chunking"],
                    },
                    "status_counts": {"oom": 3},
                    "all_validations_passed": None,
                    "mean_speedup_vs_baseline": None,
                },
                {
                    "candidate": "functional_sequential",
                    "taxonomy": {
                        "components": ["functional_call"],
                    },
                    "status_counts": {"ok": 3},
                    "all_validations_passed": False,
                    "mean_speedup_vs_baseline": 1.2,
                },
            ],
            "runs": [
                _run("baseline_original", 0, 10.0),
                _run("baseline_original", 1, 10.0),
                _run("baseline_original", 2, 10.0),
                _run("vmapped_chunk_32", 0, 9.5, True),
                _run("vmapped_chunk_32", 1, 9.4, True),
                _run("vmapped_chunk_32", 2, 9.3, True),
                _run("vmapped_chunk_64", 0, 9.2, True),
                _run("vmapped_chunk_64", 1, 9.1, True),
                _run("vmapped_chunk_64", 2, 9.0, True),
                _run("functional_sequential", 0, 8.0, False),
            ],
        }

        best = _best_valid_candidate(summary)

        self.assertIsNotNone(best)
        self.assertEqual("vmapped_chunk_64", best["candidate"])
        self.assertEqual(
            {
                "threshold_speedup": 1.05,
                "met": True,
                "candidate": "vmapped_chunk_64",
                "paired_speedup_mean": 1.098989577250447,
                "paired_speedup_min": 1.0869565217391306,
                "all_repeats_beat_baseline": True,
            },
            _acceptance(summary),
        )

    def test_acceptance_fails_without_valid_threshold_candidate(self) -> None:
        summary = {
            "candidate_summary": [
                {
                    "candidate": "vmapped_chunk_32",
                    "taxonomy": {
                        "components": ["functional_call", "vmap", "point_chunking"],
                    },
                    "status_counts": {"ok": 3},
                    "all_validations_passed": True,
                    "mean_speedup_vs_baseline": 1.02,
                }
            ],
            "runs": [
                _run("baseline_original", 0, 10.0),
                _run("baseline_original", 1, 10.0),
                _run("baseline_original", 2, 10.0),
                _run("vmapped_chunk_32", 0, 9.8, True),
                _run("vmapped_chunk_32", 1, 9.8, True),
                _run("vmapped_chunk_32", 2, 9.8, True),
            ],
        }

        self.assertEqual(
            {
                "threshold_speedup": 1.05,
                "met": False,
                "candidate": "vmapped_chunk_32",
                "paired_speedup_mean": 1.0204081632653061,
                "paired_speedup_min": 1.0204081632653061,
                "all_repeats_beat_baseline": True,
            },
            _acceptance(summary),
        )


def _run(
    candidate: str,
    repeat: int,
    total_grid_s: float,
    validation: bool | None = None,
) -> dict[str, object]:
    return {
        "candidate": candidate,
        "repeat": repeat,
        "status": "ok",
        "total_grid_s": total_grid_s,
        "validation": (
            None if validation is None else {"allclose": validation}
        ),
    }


if __name__ == "__main__":
    unittest.main()
