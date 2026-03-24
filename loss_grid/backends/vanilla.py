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

        for point in points:
            parameter_vector = build_parameter_vector(
                base_vector_device,
                direction_a_device,
                direction_b_device,
                point.alpha,
                point.beta,
            )

            transfer_start = time.perf_counter()
            apply_parameter_vector(context.model, parameter_vector)
            stage_breakdown.transfer_s += time.perf_counter() - transfer_start

            loss_value = evaluate_loss(
                model=context.model,
                data_loader=context.data_loader,
                device=context.device,
                precision=context.config.runtime.precision,
                num_batches=context.config.runtime.num_batches,
            )

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
