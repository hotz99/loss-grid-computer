from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import torch

from src.functional_eval.api_pipeline import (
    _check_tiny_workload,
    build_tiny_workload_request,
)
from src.functional_eval.experiment import (
    FunctionalEvalConfig,
    build_default_request,
    run_experiment,
)
from src.system_schema import DatasetSpec, MLTaskSpec
from src.workloads import WORKLOADS, WorkloadDefinition


ROW_GRU_WORKLOAD = "cifar10_row_gru_classification"


class FunctionalEvalMultiWorkloadSmokeTest(TestCase):
    def test_build_tiny_workload_request_uses_registered_workload_name(self) -> None:
        experiment_request = build_default_request(
            workload_name=ROW_GRU_WORKLOAD,
            device="cpu",
            sample_count=2,
            batch_size=2,
            resolution=2,
            scale=0.05,
        )
        request = build_tiny_workload_request(
            ROW_GRU_WORKLOAD,
            device="cpu",
            sample_count=2,
            batch_size=2,
            resolution=2,
            scale=0.05,
        )

        self.assertEqual(experiment_request, request)
        self.assertEqual(ROW_GRU_WORKLOAD, request.task.name)
        self.assertEqual("row_gru", request.task.model)
        self.assertEqual("image_classification", request.task.task)
        self.assertEqual("cross_entropy", request.task.loss)
        self.assertEqual("cifar10", request.task.dataset.name)
        self.assertEqual(2, request.task.dataset.sample_count)
        self.assertEqual(2, request.grid.resolution)
        self.assertEqual(2, request.mode.gpu_batch_size)
        self.assertEqual("cpu", request.device)

        with self.assertRaisesRegex(ValueError, "unknown workload"):
            build_tiny_workload_request("does_not_exist")

    def test_tiny_workload_probe_reports_missing_assets_as_skip(self) -> None:
        missing_name = "missing_assets_smoke"
        WORKLOADS[missing_name] = WorkloadDefinition(
            spec=MLTaskSpec(
                missing_name,
                DatasetSpec("missing", "assets/does-not-exist", (1,), 1),
                "missing_model",
                "classification",
                "cross_entropy",
                "assets/does-not-exist/checkpoint.pkl",
            ),
            build_dataset=lambda _spec, _seed: self.fail("dataset should not build"),
            build_model=lambda _spec: self.fail("model should not build"),
            compute_loss=lambda _model, _batch, _device: self.fail("loss should not run"),
        )
        try:
            passed: list[dict[str, object]] = []
            skipped: list[dict[str, object]] = []
            errors: list[dict[str, object]] = []

            _check_tiny_workload(
                torch,
                torch.device("cpu"),
                passed,
                skipped,
                errors,
                seed=1337,
                workload_name=missing_name,
            )
        finally:
            del WORKLOADS[missing_name]

        self.assertEqual([], passed)
        self.assertEqual([], errors)
        self.assertEqual(1, len(skipped))
        self.assertEqual("tiny_workload", skipped[0]["name"])
        self.assertIn("missing assets", str(skipped[0]["detail"]))
        self.assertEqual(missing_name, skipped[0]["metadata"]["workload"])
        self.assertFalse(skipped[0]["metadata"]["dataset_exists"])
        self.assertFalse(skipped[0]["metadata"]["checkpoint_exists"])

    def test_row_gru_cpu_smoke_exercises_baseline_functional_and_vmap_reporting(self) -> None:
        _skip_unless_assets_exist(self, ROW_GRU_WORKLOAD)
        request = build_tiny_workload_request(
            ROW_GRU_WORKLOAD,
            device="cpu",
            sample_count=2,
            batch_size=2,
            resolution=2,
            scale=0.05,
        )

        with (
            patch(
                "src.functional_eval.experiment._write_summary",
                return_value=Path("outputs/functional_eval/unit-summary.json"),
            ),
            patch("src.functional_eval.experiment._print_table"),
        ):
            summary = run_experiment(
                FunctionalEvalConfig(
                    request=request,
                    seed=1337,
                    repeats=1,
                    point_chunk_sizes=(2,),
                    run_label="unit",
                )
            )

        self.assertEqual(ROW_GRU_WORKLOAD, summary["config"]["workload_name"])
        self.assertEqual("image_classification", summary["config"]["task"])
        self.assertEqual("row_gru", summary["config"]["model"])
        self.assertEqual("cross_entropy", summary["config"]["loss"])
        self.assertEqual("cifar10", summary["config"]["dataset"]["name"])
        self.assertEqual(2, summary["config"]["dataset"]["sample_count"])

        runs = {run["candidate"]: run for run in summary["runs"]}
        self.assertEqual("ok", runs["baseline_original"]["status"])
        self.assertEqual(4, len(runs["baseline_original"]["records"]))
        self.assertEqual(
            ROW_GRU_WORKLOAD,
            runs["baseline_original"]["metadata"]["workload"],
        )

        sequential = runs["functional_sequential"]
        self.assertEqual("ok", sequential["status"])
        self.assertTrue(sequential["validation"]["allclose"])
        self.assertEqual(4, sequential["validation"]["point_count"])

        vmapped = runs["vmapped_chunk_2"]
        self.assertIn(vmapped["status"], {"error", "skipped", "ok", "oom"})
        if vmapped["status"] in {"error", "oom"}:
            self.assertIsInstance(vmapped["error"], str)
            self.assertNotIn("missing assets", vmapped["error"])


def _skip_unless_assets_exist(test_case: TestCase, workload_name: str) -> None:
    spec = WORKLOADS[workload_name].spec
    dataset_path = Path(spec.dataset.path)
    checkpoint_path = Path(spec.checkpoint_path) if spec.checkpoint_path else None
    missing = []
    if not dataset_path.exists():
        missing.append(f"dataset {dataset_path}")
    if checkpoint_path is not None and not checkpoint_path.exists():
        missing.append(f"checkpoint {checkpoint_path}")
    if missing:
        test_case.skipTest(f"missing workload assets: {', '.join(missing)}")


if __name__ == "__main__":
    import unittest

    unittest.main()
