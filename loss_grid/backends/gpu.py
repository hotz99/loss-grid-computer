from __future__ import annotations

import time

from loss_grid.backends.base import BaseLossGridExecutor
from loss_grid.config import ExperimentConfig
from loss_grid.instrumentation import StageBreakdown


class GpuLossGridExecutor(BaseLossGridExecutor):
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
            surface[point.row, point.col] = self._evaluate_point_on_device(
                context=context,
                alpha=point.alpha,
                beta=point.beta,
                base_vector_device=base_vector_device,
                direction_a_device=direction_a_device,
                direction_b_device=direction_b_device,
                stage_breakdown=stage_breakdown,
            )
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
