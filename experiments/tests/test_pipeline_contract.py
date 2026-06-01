from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import runner


_WORKLOAD = "mnist_mlp_classification"


def _assets_present() -> bool:
    return Path("assets/mnist/MNIST/raw/t10k-images-idx3-ubyte").exists()


def _run_tiny_suite(out_dir: str) -> Path:
    """Drive the real runner end to end on a deliberately tiny CPU config.

    Forces CPU (device-agnostic orchestration), one cheap workload, a 2x2 grid,
    and R=2 so the paired t-CI path produces real claim statuses rather than
    insufficient_data. Compile candidates are off to keep wall time bounded;
    candidate-level numerical correctness lives in test_surface_equivalence.
    """
    overrides = {
        "OUTPUT_DIR": out_dir,
        "DEVICE": "cpu",
        "WORKLOAD_NAMES": (_WORKLOAD,),
        "SAMPLE_COUNT": 64,
        "GRID_RESOLUTION": 2,
        "EXPERIMENT_3_SESSION_GRID_RESOLUTION": 3,
        "GPU_BATCH_SIZE": 32,
        "REPEATS": 2,
        "SLOWDOWN_CEILING": 2,
        "POINT_CHUNK_SIZES": (32,),
        "INCLUDE_COMPILE_CANDIDATES": False,
        "MAX_CPU_WORKER_CANDIDATE": 1,
        "RUN_PLATFORM_INVENTORY": False,
        "RUN_EXPERIMENT_1": True,
        "RUN_EXPERIMENT_2": True,
        "RUN_EXPERIMENT_3": True,
        "RUN_PROJECTION": True,
        "REUSE_EXPERIMENT_1_FROM": None,
        "REUSE_EXPERIMENT_2_FROM": None,
    }
    with patch.object(runner, "_banner", lambda _name, _marker: None):
        with _patched(overrides):
            return runner.main()


def _patched(overrides):
    from contextlib import ExitStack

    stack = ExitStack()
    for name, value in overrides.items():
        stack.enter_context(patch.object(runner, name, value))
    return stack


class RunnerPipelineContractTest(unittest.TestCase):
    """End-to-end contract guard for the experiment runner.

    Validates the orchestration and the data contract the thesis tables read
    from: schema versions, the exp1 -> exp3 rq3_config wiring, projection
    composition, and that R>=2 yields CI-bounded claim statuses. It does not
    assert specific verdicts (those depend on the device and the run).
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not _assets_present():
            raise unittest.SkipTest("MNIST assets not present")
        cls._tmp = tempfile.TemporaryDirectory()
        run_dir = _run_tiny_suite(cls._tmp.name)
        cls._artifacts = {
            path.stem: json.loads(path.read_text())
            for path in Path(run_dir).glob("*.json")
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_all_artifacts_written(self) -> None:
        for name in ("config", "experiment-1", "experiment-2", "experiment-3",
                     "projection", "suite"):
            self.assertIn(name, self._artifacts, f"missing artifact {name}.json")

    def test_config_records_enabled_toggles(self) -> None:
        toggles = self._artifacts["config"]["run_toggles"]
        for step in ("experiment_1", "experiment_2", "experiment_3", "projection"):
            self.assertTrue(toggles[step], f"{step} should be enabled")

    def test_experiment_1_contract(self) -> None:
        exp1 = self._artifacts["experiment-1"]
        self.assertEqual("completed", exp1["status"])
        self.assertEqual("experiment-1-algorithm-v1", exp1["schema_version"])
        record = exp1["record"]
        self.assertIsInstance(record["rq3_config"], str)
        self.assertIn(_WORKLOAD, record["rq3_config_by_workload"])
        workload = record["workloads"][_WORKLOAD]
        candidates = {c["candidate"]: c for c in workload["candidates"]}
        self.assertIn("vmapped_k32", candidates)
        for cand in candidates.values():
            self.assertIn("claim_status", cand)
            self.assertIn("speedup_mean", cand)
        runner_validation = record["runner_surface_validation"]
        self.assertTrue(runner_validation["valid"])
        self.assertGreater(runner_validation["surface_pair_count"], 0)
        composition = exp1["composition"]["per_workload"][_WORKLOAD]
        self.assertIn("composition_status", composition)

    def test_repeats_two_yields_ci_bounded_claims(self) -> None:
        # Guards the R=1 -> insufficient_data blocker: with R>=2 and a valid
        # surface, the paired t-CI must resolve to a real claim status.
        workload = self._artifacts["experiment-1"]["record"]["workloads"][_WORKLOAD]
        vmapped = next(
            c for c in workload["candidates"] if c["candidate"] == "vmapped_k32"
        )
        self.assertNotEqual("insufficient_data", vmapped["claim_status"])
        self.assertIn(
            vmapped["claim_status"], {"speedup", "regression", "inconclusive"}
        )
        self.assertIsNotNone(vmapped["speedup_ci_low"])
        self.assertIsNotNone(vmapped["speedup_ci_high"])

    def test_experiment_2_contract(self) -> None:
        exp2 = self._artifacts["experiment-2"]
        self.assertEqual("completed", exp2["status"])
        self.assertEqual("experiment-2-hybrid-v1", exp2["schema_version"])
        workload = exp2["record"]["workloads"][_WORKLOAD]
        self.assertEqual("completed", workload["status"])
        for key in ("regime_predictor", "ladder", "threshold_status",
                    "threshold_slowdown"):
            self.assertIn(key, workload)
        # The ladder is swept in full from slow=1, independent of r_native.
        self.assertEqual(1, workload["ladder"][0]["slowdown_factor"])
        for rung in workload["ladder"]:
            self.assertIn("claim_status", rung)
            self.assertIn(
                rung["claim_status"],
                {"hybrid_wins", "inconclusive", "hybrid_regresses", "invalid_surface"},
            )
        runner_validation = exp2["record"]["runner_surface_validation"]
        self.assertTrue(runner_validation["valid"])
        self.assertGreater(runner_validation["surface_pair_count"], 0)

    def test_experiment_3_contract(self) -> None:
        exp3 = self._artifacts["experiment-3"]
        self.assertEqual("experiment-3-composition-v1", exp3["schema_version"])
        self.assertEqual("completed", exp3["status"])
        record = exp3["record"]
        runner_validation = record["runner_surface_validation"]
        self.assertTrue(runner_validation["valid"])
        self.assertGreater(runner_validation["surface_pair_count"], 0)
        # The composition sweep reports one cell per workload, no affinity filter.
        cells = record["cells"]
        self.assertIn(_WORKLOAD, cells)
        cell = cells[_WORKLOAD]
        self.assertIn(
            cell["status"], {"completed", "skipped", "selection_starvation"}
        )
        self.assertNotIn("achieved_r", cell["cell"])
        self.assertNotIn("slowdown_factor", cell["cell"])
        if cell["status"] == "completed":
            self.assertIn(
                cell["composition_verdict"], {"complement", "dominate", "surface_invalid"}
            )
            self.assertIn("q_cross", cell)
            self.assertIn("selected_path", cell)
            self.assertIn("selection_probe_s", cell)
            self.assertIn("N_star_compile", cell)
            self.assertIn("compile_reuse_label", cell)
            # Retired RQ3 setup-recovery fields must be gone.
            self.assertNotIn("break_even_hybrid", cell)
            self.assertNotIn("hybrid_label", cell)

    def test_projection_wires_rq3_config_from_experiment_1(self) -> None:
        projection = self._artifacts["projection"]
        self.assertEqual("paper-projection-v3", projection["schema_version"])
        self.assertEqual(
            self._artifacts["experiment-1"]["record"]["rq3_config"],
            projection["rq1"]["rq3_config"],
        )
        claims = projection["claims"]
        for ready in ("rq1_ready", "rq2_ready", "rq3_ready"):
            self.assertTrue(claims[ready], f"{ready} should be true")
        rq3_cell = projection["rq3"]["cells"][0]
        self.assertNotIn("achieved_r", rq3_cell)
        self.assertNotIn("slowdown_factor", rq3_cell)

    def test_suite_contract(self) -> None:
        suite = self._artifacts["suite"]
        self.assertEqual("platform-experiment-suite-v2", suite["schema_version"])
        # A healthy run is "planned" (the projection step is a paper-projection
        # scaffold that is always "planned"); the failure signal is errors or
        # unknown workloads, which surface as "completed_with_errors".
        self.assertNotEqual("completed_with_errors", suite["status"])
        self.assertIn(suite["status"], {"completed", "planned"})
        records = suite["records"]
        for step in ("experiment_1", "experiment_2", "experiment_3", "projection"):
            self.assertIn(step, records)
            self.assertNotEqual("error", records[step].get("status"))


if __name__ == "__main__":
    unittest.main()
