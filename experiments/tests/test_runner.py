from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import runner


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


if __name__ == "__main__":
    unittest.main()
