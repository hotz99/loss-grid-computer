from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import runner
from experiments.schemas import (
    CandidateRunResult,
    Experiment1Config,
    Experiment1Result,
)


class ExperimentRunnerTest(unittest.TestCase):
    def test_main_writes_disabled_suite_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(runner, "OUTPUT_DIR", tmpdir),
                patch.object(runner, "RUN_LABEL", "runner regression"),
                patch.object(runner, "RUN_PLATFORM_INVENTORY", False),
                patch.object(runner, "RUN_EXPERIMENT_1", False),
                patch.object(runner, "RUN_EXPERIMENT_2", False),
                patch.object(runner, "RUN_EXPERIMENT_3", False),
                patch.object(runner, "RUN_PROJECTION", False),
                patch.object(runner, "REUSE_EXPERIMENT_1_FROM", None),
                patch.object(runner, "REUSE_EXPERIMENT_2_FROM", None),
                patch.object(runner, "_banner", lambda _name, _marker: None),
            ):
                run_dir = runner.main()

            output_dir = Path(tmpdir)
            self.assertEqual(output_dir, run_dir)

            config = json.loads((output_dir / "config.json").read_text())
            suite = json.loads((output_dir / "suite.json").read_text())
            experiment_1 = json.loads((output_dir / "experiment-1.json").read_text())
            projection = json.loads((output_dir / "projection.json").read_text())

        self.assertEqual(
            {
                "platform_inventory": False,
                "experiment_1": False,
                "experiment_2": False,
                "experiment_3": False,
                "projection": False,
            },
            config["run_toggles"],
        )
        self.assertEqual("platform-experiment-suite-v2", suite["schema_version"])
        self.assertEqual("completed", suite["status"])
        self.assertEqual("disabled", experiment_1["status"])
        self.assertEqual("disabled", suite["records"]["experiment_1"]["status"])
        self.assertEqual("disabled", projection["status"])
        self.assertEqual("projection.json", suite["artifacts"]["projection"])

    def test_run_step_fails_fast_on_invalid_surface_pair(self) -> None:
        result = Experiment1Result(
            status="completed",
            schema_version="experiment-1-algorithm-v1",
            config=Experiment1Config(workload_names=("toy",), repeats=1),
            trials=(),
            runs=(
                CandidateRunResult(
                    workload_name="toy",
                    candidate="baseline",
                    role="baseline",
                    repeat=0,
                    status="ok",
                    trial_order=(),
                    records=((0, 0, 1.0),),
                ),
                CandidateRunResult(
                    workload_name="toy",
                    candidate="vmapped_k32",
                    role="vmapped",
                    repeat=0,
                    status="ok",
                    trial_order=(),
                    records=((0, 0, 2.0),),
                ),
            ),
            aggregates=(),
            rq3_config="baseline",
            composition={},
            record={"status": "completed"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiment-1.json"
            with (
                patch.object(runner, "FAIL_FAST", True),
                patch.object(runner, "_banner", lambda _name, _marker: None),
                self.assertRaises(runner.PipelineContractError),
            ):
                runner._run_step("experiment_1", path, lambda: result)

            error_payload = json.loads(path.read_text())

        self.assertEqual("error", error_payload["status"])
        self.assertIn("failed surface gate", error_payload["error"])

    def test_run_step_records_runner_surface_validation(self) -> None:
        result = Experiment1Result(
            status="completed",
            schema_version="experiment-1-algorithm-v1",
            config=Experiment1Config(workload_names=("toy",), repeats=1),
            trials=(),
            runs=(
                CandidateRunResult(
                    workload_name="toy",
                    candidate="baseline",
                    role="baseline",
                    repeat=0,
                    status="ok",
                    trial_order=(),
                    records=((0, 0, 1.0),),
                ),
                CandidateRunResult(
                    workload_name="toy",
                    candidate="vmapped_k32",
                    role="vmapped",
                    repeat=0,
                    status="ok",
                    trial_order=(),
                    records=((0, 0, 1.0),),
                ),
            ),
            aggregates=(),
            rq3_config="baseline",
            composition={},
            record={"status": "completed"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiment-1.json"
            with patch.object(runner, "_banner", lambda _name, _marker: None):
                runner._run_step("experiment_1", path, lambda: result)
            payload = json.loads(path.read_text())

        validation = payload["record"]["runner_surface_validation"]
        self.assertTrue(validation["valid"])
        self.assertEqual(1, validation["surface_pair_count"])


if __name__ == "__main__":
    unittest.main()
