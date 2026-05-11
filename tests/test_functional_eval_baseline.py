from __future__ import annotations

import unittest

import torch
from torch.utils.data import TensorDataset

from src.functional_eval.baseline import run_baseline
from src.system_schema import DatasetSpec, GridSpec, MLTaskSpec, SchedulerRequest, VanillaMode
from src.workloads import WORKLOADS, WorkloadDefinition


class FunctionalEvalBaselineTest(unittest.TestCase):
    def test_runs_generic_workload_compute_loss_contract(self) -> None:
        workload_name = "test_tiny_baseline_regression"
        calls: list[int] = []

        def build_dataset(_spec: MLTaskSpec, _seed: int) -> TensorDataset:
            inputs = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                dtype=torch.float32,
            )
            targets = torch.tensor([0.0, 1.0, 1.0, 2.0], dtype=torch.float32)
            return TensorDataset(inputs, targets)

        def build_model(_spec: MLTaskSpec) -> torch.nn.Module:
            return torch.nn.Linear(2, 1)

        def compute_loss(
            model: torch.nn.Module,
            batch: tuple[torch.Tensor, torch.Tensor],
            device: torch.device,
        ) -> tuple[torch.Tensor, int]:
            calls.append(1)
            inputs, targets = batch
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            predictions = model(inputs).squeeze(-1)
            return torch.nn.MSELoss()(predictions, targets), int(targets.shape[0])

        task = MLTaskSpec(
            workload_name,
            DatasetSpec("tensor", "", (2,), 4),
            "linear_regressor",
            "tabular_regression",
            "mse",
            None,
        )
        previous = WORKLOADS.get(workload_name)
        WORKLOADS[workload_name] = WorkloadDefinition(
            spec=task,
            build_dataset=build_dataset,
            build_model=build_model,
            compute_loss=compute_loss,
        )
        try:
            result = run_baseline(
                SchedulerRequest(
                    task=task,
                    grid=GridSpec(resolution=2, scale=0.1),
                    mode=VanillaMode(gpu_batch_size=2),
                    device="cpu",
                ),
                seed=7,
            )
        finally:
            if previous is None:
                del WORKLOADS[workload_name]
            else:
                WORKLOADS[workload_name] = previous

        self.assertEqual("baseline_original", result.candidate)
        self.assertEqual(4, len(result.records))
        self.assertEqual(8, len(calls))
        self.assertEqual("generic vanilla point-loop semantics", result.metadata["wrapped"])
        self.assertEqual(workload_name, result.metadata["workload"])
        self.assertGreaterEqual(result.timings.perturbation_s, 0.0)
        self.assertGreaterEqual(result.timings.binding_s, 0.0)
        self.assertGreaterEqual(result.timings.batch_eval_s, 0.0)
        self.assertGreaterEqual(result.timings.total_grid_s, 0.0)


if __name__ == "__main__":
    unittest.main()
