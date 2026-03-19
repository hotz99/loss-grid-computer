from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Sequence

import torch

from loss_grid.config import ExperimentConfig
from loss_grid.grid import GridPoint, partition_points
from loss_grid.instrumentation import StageBreakdown
from loss_grid.kernel import (
    apply_parameter_vector,
    build_execution_context,
    build_parameter_vector,
    evaluate_loss,
)
from loss_grid.metrics import build_metric_record
from loss_grid.results import ExperimentResult


class LossGridExecutor(ABC):
    @abstractmethod
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        raise NotImplementedError


class BaseLossGridExecutor(LossGridExecutor):
    def _output_dir(self, config: ExperimentConfig) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return str(Path(config.runtime.output_root) / f"{config.experiment_name}-{timestamp}")

    def _make_surface(self, resolution: int) -> torch.Tensor:
        return torch.full((resolution, resolution), float("nan"), dtype=torch.float32)

    def _common_setup(self, config: ExperimentConfig):
        context = build_execution_context(config)
        surface = self._make_surface(config.grid.resolution)
        return context, surface

    def _partition(self, config: ExperimentConfig, points: Sequence[GridPoint], rank: int, workers: int) -> List[GridPoint]:
        return partition_points(points, config.grid, config.decomposition, rank=rank, workers=workers)

    def _finalize_result(
        self,
        config: ExperimentConfig,
        surface: torch.Tensor,
        stage_breakdown: StageBreakdown,
        environment: Dict,
        device_name: str,
        rank: int,
        world_size: int,
        output_dir: str,
        is_root: bool = True,
    ) -> ExperimentResult:
        metrics = build_metric_record(
            num_points=int(surface.numel()),
            total_runtime_s=stage_breakdown.total_s,
            baseline_time_s_strong=config.reference.strong_scaling.baseline_time_s,
            baseline_time_s_weak=config.reference.weak_scaling.baseline_time_s,
            workers=world_size,
        )
        runtime_log = {
            "stage_breakdown": {
                "perturbation_s": stage_breakdown.perturbation_s,
                "transfer_s": stage_breakdown.transfer_s,
                "forward_s": stage_breakdown.forward_s,
                "communication_s": stage_breakdown.communication_s,
                "gpu_kernel_s": stage_breakdown.gpu_kernel_s,
                "host_preprocessing_s": stage_breakdown.host_preprocessing_s,
                "total_s": stage_breakdown.total_s,
                "overlap_efficiency": stage_breakdown.overlap_efficiency,
            }
        }
        cpu_workers = int(config.resources.cpu_workers) if config.backend.lower() == "hybrid" else 0
        return ExperimentResult(
            experiment_name=config.experiment_name,
            output_dir=output_dir,
            backend=config.backend,
            device={
                "gpu": device_name,
                "cpu": cpu_workers,
            },
            rank=rank,
            world_size=world_size,
            surface=surface if is_root else None,
            stage_breakdown=stage_breakdown,
            metrics=metrics,
            config=config.to_dict(),
            environment=environment,
            runtime_log=runtime_log,
            is_root=is_root,
        )

    def _evaluate_point_on_device(
        self,
        context,
        alpha: float,
        beta: float,
        base_vector_device: torch.Tensor,
        direction_a_device: torch.Tensor,
        direction_b_device: torch.Tensor,
        stage_breakdown: StageBreakdown,
    ) -> float:
        point_start = time.perf_counter()
        perturb_start = time.perf_counter()
        parameter_vector = build_parameter_vector(
            base_vector_device,
            direction_a_device,
            direction_b_device,
            alpha,
            beta,
        )
        stage_breakdown.perturbation_s += time.perf_counter() - perturb_start

        transfer_start = time.perf_counter()
        apply_parameter_vector(context.model, parameter_vector)
        stage_breakdown.transfer_s += time.perf_counter() - transfer_start

        forward_start = time.perf_counter()
        loss_value, gpu_kernel_s = evaluate_loss(
            model=context.model,
            data_loader=context.data_loader,
            device=context.device,
            precision=context.config.runtime.precision,
            num_batches=context.config.runtime.num_batches,
        )
        stage_breakdown.forward_s += time.perf_counter() - forward_start
        stage_breakdown.gpu_kernel_s += gpu_kernel_s

        slowdown_factor = float(getattr(context.config.runtime, "gpu_slowdown_factor", 1.0))
        if context.device.type != "cpu" and slowdown_factor > 1.0:
            point_elapsed = time.perf_counter() - point_start
            extra_delay = point_elapsed * (slowdown_factor - 1.0)
            time.sleep(extra_delay)
        return loss_value
