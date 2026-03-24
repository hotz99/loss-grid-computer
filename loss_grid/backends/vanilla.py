from __future__ import annotations

import time

from loss_grid.backends.base import BaseLossGridExecutor
from loss_grid.config import ExperimentConfig
from loss_grid.instrumentation import StageBreakdown
from loss_grid.kernel import (
    apply_parameter_vector,
    build_parameter_vector,
    evaluate_loss,
)


class VanillaGpuLossGridExecutor(BaseLossGridExecutor):
    """Single-process canonical baseline: point-by-point eager evaluation."""

    def run(self, config: ExperimentConfig):
        context, surface = self._common_setup(config)
        stage_breakdown = StageBreakdown()
        points = self._partition(config, context.points, rank=0, workers=1)
        output_dir = self._output_dir(config)

        base_vector_device = context.base_vector_cpu.to(context.device)
        direction_a_device = context.direction_a_cpu.to(context.device)
        direction_b_device = context.direction_b_cpu.to(context.device)

        total_start = time.perf_counter()

        print(f"HERE WE ARE")
        for point in points:
            perturb_start = time.perf_counter()
            parameter_vector = build_parameter_vector(
                base_vector_device,
                direction_a_device,
                direction_b_device,
                point.alpha,
                point.beta,
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

            surface[point.row, point.col] = loss_value

        total_runtime = time.perf_counter() - total_start
        stage_breakdown.finalize(total_runtime)
        return self._finalize_result(
            config=config,
            surface=surface,
            stage_breakdown=stage_breakdown,
            environment=context.environment,
            device_name=str(context.device),
            rank=0,
            world_size=1,
            output_dir=output_dir,
            is_root=True,
        )
