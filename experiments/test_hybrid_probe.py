from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments import hybrid_applicability
from experiments.primitives import hybrid_probe
from src.results import DeviceRecord, ExperimentResult, Measurement, RunRecord
from src.schemas import DatasetSpec, GridSpec, HybridMode, MLTaskSpec, SchedulerRequest


class HybridProbeTest(unittest.TestCase):
    def test_repeats_reuse_surface_seed(self) -> None:
        seeds: list[int] = []

        def fake_run_backend(
            request: SchedulerRequest,
            *,
            seed: int,
            gpu_slowdown_factor: float,
        ) -> SimpleNamespace:
            del request
            seeds.append(seed)
            hybrid_total_s = 2.0 / gpu_slowdown_factor
            return SimpleNamespace(
                record=SimpleNamespace(
                    experiment_name="probe",
                    measurement=SimpleNamespace(total_s=hybrid_total_s),
                ),
                records=[(0, 0, 1.0)],
            )

        request = SchedulerRequest(
            task=MLTaskSpec(
                "probe",
                DatasetSpec("dummy", "assets", (1,), 1),
                "dummy_model",
                "dummy_task",
                "mse",
            ),
            grid=GridSpec(1, 1.0),
            mode=HybridMode(8, 4, 1),
            device="cpu",
        )

        with patch.object(hybrid_probe, "run_backend", side_effect=fake_run_backend):
            result = hybrid_probe.main(
                hybrid_request=request,
                vanilla_surface=[(0, 0, 1.0)],
                vanilla_total_s=1.0,
                bracket_repeats=2,
                sample_repeats=3,
                max_slowdown=2.0,
                jump_factor=2.0,
                linear_samples=2,
                atol=1e-6,
                rtol=1e-5,
                seed=1337,
                cpu_workers=1,
                cpu_batch_size=4,
            )

        self.assertEqual("crossover_found", result["status"])
        self.assertGreater(len(seeds), 0)
        self.assertEqual({1337}, set(seeds))

    def test_slowdown_adjusted_vanilla_result_is_derived_from_profiled_baseline(self) -> None:
        native = ExperimentResult(
            record=RunRecord(
                experiment_name="probe",
                measurement=Measurement(total_s=10.0, num_points=4),
                backend="vanilla",
                device=DeviceRecord("cpu", 0),
                config={},
                comparison=None,
                output_dir="",
            ),
            runtime_log={
                "total_s": 10.0,
                "vanilla_execution": {
                    "grid_compute_only_s": 10.0,
                    "throughput_points_per_s": 0.4,
                },
            },
            records=[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)],
        )

        slowed = hybrid_applicability._slowdown_adjusted_vanilla_result(native, 2.5)

        self.assertEqual(25.0, slowed.record.measurement.total_s)
        self.assertEqual(4, slowed.record.measurement.num_points)
        self.assertEqual(0.16, slowed.runtime_log["vanilla_execution"]["throughput_points_per_s"])
        self.assertEqual("derived_from_profiled_vanilla", slowed.runtime_log["vanilla_execution"]["slowdown_source"])
        self.assertEqual(native.records, slowed.records)


if __name__ == "__main__":
    unittest.main()
