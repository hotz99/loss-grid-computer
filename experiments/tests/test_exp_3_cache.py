from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments import composition_selection as selection_mod
from experiments.composition_selection import CompositionSelection
from experiments.candidates import GpuCandidate
from experiments.candidates.base import CandidateRunOutput
from experiments.exp_3_cache import (
    _compile_reuse_label,
    _composition_verdict,
    _measure_compile_cost,
    _n_star_compile,
    _probe_grid_for,
    _r_native_for,
    _run_cell,
    _sweep_workloads,
)
from experiments.schemas import (
    DatasetSpec,
    Experiment1Config,
    Experiment1Result,
    Experiment2Config,
    Experiment2Result,
    Experiment3Config,
    GridSpec,
    MLTaskSpec,
)
from experiments.sessions import gpu_only_session


def _exp1(
    rq3_config_by_workload: dict,
    rq3_config: str = "compiled_vmapped_k64",
) -> Experiment1Result:
    return Experiment1Result(
        status="completed",
        schema_version="experiment-1-algorithm-v1",
        config=Experiment1Config(),
        trials=(),
        runs=(),
        aggregates=(),
        rq3_config=rq3_config,
        composition={},
        record={"rq3_config_by_workload": rq3_config_by_workload},
    )


def _exp2(workloads: dict) -> Experiment2Result:
    return Experiment2Result(
        status="completed",
        schema_version="experiment-2-hybrid-v1",
        config=Experiment2Config(),
        result={},
        record={"workloads": workloads},
    )


class CompositionVerdictTest(unittest.TestCase):
    def test_q_cross_above_one_complements(self) -> None:
        self.assertEqual("complement", _composition_verdict(1.4))

    def test_q_cross_at_or_below_one_is_dominated(self) -> None:
        # gpu_only sets q_cross = 1: A dominates B, a result.
        self.assertEqual("dominate", _composition_verdict(1.0))
        self.assertEqual("dominate", _composition_verdict(0.7))

    def test_suppressed_surface_has_no_verdict(self) -> None:
        self.assertEqual("surface_invalid", _composition_verdict(None))


class NStarCompileTest(unittest.TestCase):
    def test_compile_break_even_folds_cold_start(self) -> None:
        # compile 6s, per-variant saving T_vanilla - T_gpu_only = 2 -> 3 variants.
        self.assertEqual(3, _n_star_compile(6.0, 10.0, 8.0, compiles=True))

    def test_baseline_cell_is_undefined(self) -> None:
        self.assertIsNone(_n_star_compile(0.0, 10.0, 8.0, compiles=False))

    def test_infinite_when_compiled_not_faster(self) -> None:
        self.assertEqual(float("inf"), _n_star_compile(6.0, 8.0, 8.0, compiles=True))

    def test_three_way_component_labels(self) -> None:
        self.assertEqual("supported", _compile_reuse_label(4, compiles=True))
        self.assertEqual(
            "supported_asymptotically",
            _compile_reuse_label(5, compiles=True),
        )
        self.assertEqual("refuted", _compile_reuse_label(float("inf"), compiles=True))
        self.assertEqual("undefined", _compile_reuse_label(None, compiles=False))


class SweepInputsTest(unittest.TestCase):
    def test_sweep_uses_rq1_per_workload_cells(self) -> None:
        exp1 = _exp1({"w_a": "compiled_vmapped_k32", "w_b": "baseline"})
        exp2 = _exp2({})
        self.assertEqual(("w_a", "w_b"), _sweep_workloads(exp1, exp2))

    def test_sweep_falls_back_to_exp2_workloads(self) -> None:
        exp1 = _exp1({})
        exp2 = _exp2({"w_c": {"status": "completed"}})
        self.assertEqual(("w_c",), _sweep_workloads(exp1, exp2))

    def test_r_native_read_from_exp2_predictor(self) -> None:
        exp2 = _exp2(
            {"w_a": {"status": "completed", "regime_predictor": {"r_native": 2.5}}}
        )
        self.assertEqual(2.5, _r_native_for(exp2, "w_a"))
        self.assertIsNone(_r_native_for(exp2, "missing"))


class ProbeGridTest(unittest.TestCase):
    def test_probe_grid_accounts_for_gpu_chunk_and_cpu_workers(self) -> None:
        self.assertEqual(9, _probe_grid_for(2, 64).resolution)
        self.assertEqual(3, _probe_grid_for(2, 1).resolution)


class SelectionStarvationGuardTest(unittest.TestCase):
    def test_selection_records_max_hybrid_cpu_points(self) -> None:
        task = MLTaskSpec(
            name="unit",
            dataset=DatasetSpec("unit", "unused", (2,), 1),
            model="unit",
            task="regression",
            loss="mse",
        )
        outputs = iter(
            [
                CandidateRunOutput(
                    records=[],
                    total_grid_s=9.0,
                    worker_throughput_split={"cpu_points": 0},
                ),
                CandidateRunOutput(
                    records=[],
                    total_grid_s=8.0,
                    worker_throughput_split={"cpu_points": 3},
                ),
            ]
        )
        with patch(
            "experiments.composition_selection.hybrid.run",
            lambda *_args, **_kwargs: next(outputs),
        ):
            selection = selection_mod.select_composition(
                task,
                GridSpec(3, 1.0),
                gpu_batch_size=1,
                baseline_total_s=10.0,
                cpu_workers=(1,),
                cpu_batch_sizes=(1, 2),
                patience=3,
                device=torch.device("cpu"),
                seed=0,
            )
        self.assertEqual("gpu_cpu_hybrid", selection.selected_path)
        self.assertEqual(3, selection.max_hybrid_cpu_points)

    def test_zero_cpu_probe_work_starves_even_when_gpu_only_selected(self) -> None:
        workload = "mnist_mlp_classification"
        task = MLTaskSpec(
            name=workload,
            dataset=DatasetSpec("unit", "unused", (2,), 4),
            model="unit",
            task="classification",
            loss="cross_entropy",
        )
        selection = CompositionSelection(
            selected_path="gpu_only",
            gpu_batch_size=1,
            cpu_batch_size=None,
            cpu_workers=None,
            selection_probe_s=1.0,
            baseline_total_s=1.0,
            selected_total_s=None,
            max_hybrid_cpu_points=0,
        )
        with patch("experiments.exp_3_cache.task_for_workload", return_value=task), \
             patch(
                 "experiments.exp_3_cache.list_same_family_checkpoints",
                 return_value=("ckpt-0", "ckpt-1", "ckpt-2", "ckpt-3"),
             ), \
             patch(
                 "experiments.exp_3_cache.selection_mod.cpu_worker_candidates",
                 return_value=(1,),
             ), \
             patch(
                 "experiments.exp_3_cache.selection_mod.cpu_batch_size_candidates",
                 return_value=(1,),
             ), \
             patch(
                 "experiments.exp_3_cache.run_standalone",
                 return_value=SimpleNamespace(total_grid_s=1.0),
             ), \
             patch(
                 "experiments.exp_3_cache.selection_mod.select_composition",
                 return_value=selection,
             ), \
             patch("experiments.exp_3_cache._measure_compile_cost") as measure:
            record, pairs = _run_cell(
                Experiment3Config(sample_count=4, gpu_batch_size=1),
                _exp1({workload: "baseline"}, rq3_config="baseline"),
                _exp2({workload: {"regime_predictor": {"r_native": 2.0}}}),
                workload,
                torch.device("cpu"),
            )

        self.assertEqual("selection_starvation", record["status"])
        self.assertEqual("selection_starvation", record["composition_verdict"])
        self.assertEqual([], pairs)
        measure.assert_not_called()


class CompositionSelectionScoreTest(unittest.TestCase):
    def test_composition_selection_uses_steady_state_not_spawn_inclusive_time(self) -> None:
        task = MLTaskSpec(
            name="unit",
            dataset=DatasetSpec("unit", "unused", (2,), 1),
            model="unit",
            task="regression",
            loss="mse",
        )
        output = CandidateRunOutput(
            records=[],
            total_grid_s=9.0,
            worker_throughput_split={
                "cpu_points": 2,
                "gpu_points": 3,
                "cpu_max_wall_s": 2.0,
                "gpu_wall_s": 3.0,
            },
        )
        with patch("experiments.composition_selection.hybrid.run", return_value=output):
            selection = selection_mod.select_composition(
                task,
                GridSpec(3, 1.0),
                gpu_batch_size=1,
                baseline_total_s=5.0,
                cpu_workers=(1,),
                cpu_batch_sizes=(1,),
                patience=3,
                device=torch.device("cpu"),
                seed=0,
            )

        self.assertEqual("gpu_cpu_hybrid", selection.selected_path)
        self.assertEqual(9.0, selection.selection_probe_s)
        self.assertEqual(3.0, selection.selected_total_s)
        self.assertEqual(
            3.0,
            selection.selection_trials[0]["steady_state_selection_total_s"],
        )
        self.assertEqual(9.0, selection.selection_trials[0]["spawn_inclusive_total_s"])


class MeasureCompileCostTest(unittest.TestCase):
    def test_non_compiling_roles_cost_zero_without_running(self) -> None:
        # baseline/vmapped never compile: the helper must short-circuit to 0.0
        # so the one-time setup cost stays uniform across workloads, and it must
        # not invoke run_standalone (no task/device needed here).
        for candidate in (GpuCandidate.baseline(), GpuCandidate.vmapped(32)):
            cost = _measure_compile_cost(
                candidate, task=None, grid=None,
                gpu_batch_size=32, device=None, seed=0,
            )
            self.assertEqual(0.0, cost)


class CompositionSelectionCellTest(unittest.TestCase):
    def test_selection_cell_carries_probe_time(self) -> None:
        selection = CompositionSelection(
            selected_path="gpu_only", gpu_batch_size=32, cpu_batch_size=None,
            cpu_workers=None, selection_probe_s=4.0, baseline_total_s=10.0,
            selected_total_s=None,
        )
        self.assertEqual(4.0, selection.selection_probe_s)


class WarmGpuOnlySessionTest(unittest.TestCase):
    def test_compiled_vmapped_session_reuses_evaluator_without_recompile(self) -> None:
        task = MLTaskSpec(
            name="unit",
            dataset=DatasetSpec("unit", "unused", (2,), 1),
            model="unit",
            task="regression",
            loss="mse",
            checkpoint_path="ckpt-0.pkl",
        )
        checkpoints = ("ckpt-0.pkl", "ckpt-1.pkl")
        evaluator = _FakeCompiledVmappedEvaluator()

        with patch("experiments.sessions.device_mod.seed_all", lambda _device, _seed: None), \
             patch("experiments.sessions.device_mod.synchronize", lambda _device: None), \
             patch("experiments.sessions.device_mod.apply_gpu_slowdown", lambda *_args: None), \
             patch("experiments.sessions.build_model", return_value=torch.nn.Linear(2, 1)), \
             patch("experiments.sessions.load_checkpoint", lambda _model, _path: None), \
             patch("experiments.sessions.build_dataset", return_value=[(torch.zeros(2), torch.zeros(1))]), \
             patch("experiments.sessions.build_dataloader", return_value=[]), \
             patch("experiments.sessions.make_chunk_evaluator", return_value=evaluator) as make_eval:
            session = gpu_only_session(
                task,
                GridSpec(3, 1.0),
                checkpoints,
                gpu_candidate=GpuCandidate.compiled_vmapped(64),
                batch_size=1,
                device=torch.device("cpu"),
                seed=0,
            )

        self.assertEqual(1, make_eval.call_count)
        self.assertEqual(1, evaluator.warmup_count)
        self.assertEqual(2, evaluator.evaluate_count)
        self.assertTrue(
            all(
                item.diagnostics.get("recompile_count") == 0
                for item in session.per_checkpoint
            )
        )


class _FakeCompiledVmappedEvaluator:
    def __init__(self) -> None:
        self.warmup_count = 0
        self.evaluate_count = 0

    def warmup(self) -> float:
        self.warmup_count += 1
        return 0.01

    def evaluate(self, chunk):
        self.evaluate_count += 1
        return [(point.row, point.col, float(self.evaluate_count)) for point in chunk]

    def diagnostics(self) -> dict:
        return {"recompile_count": 0}


if __name__ == "__main__":
    unittest.main()
