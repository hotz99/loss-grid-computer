from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Sequence

import torch
from torch.nn.utils import vector_to_parameters

from src.config import ExperimentConfig
from src.directions import build_parameter_vector
from src.grid import GridPoint
from src.kernel import (
    evaluate_loss,
    evaluate_loss_compiled_chunk,
)


def build_output_dir(config: ExperimentConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(
        Path(config.runtime.output_root) / f"{config.experiment_name}-{timestamp}"
    )


def make_surface(resolution: int) -> torch.Tensor:
    return torch.full((resolution, resolution), float("nan"), dtype=torch.float32)


def evaluate_point_on_device(
    context,
    alpha: float,
    beta: float,
    base_vector_device: torch.Tensor,
    direction_a_device: torch.Tensor,
    direction_b_device: torch.Tensor,
) -> float:
    point_start = time.perf_counter()
    parameter_vector = build_parameter_vector(
        base_vector_device,
        direction_a_device,
        direction_b_device,
        alpha,
        beta,
    )
    vector_to_parameters(parameter_vector, context.model.parameters())

    loss_value = evaluate_loss(
        model=context.model,
        data_loader=context.data_loader,
        device=context.device,
        num_batches=context.config.runtime.num_batches,
    )

    slowdown_factor = context.config.runtime.gpu_slowdown_factor
    if context.device.type != "cpu" and slowdown_factor > 1.0:
        point_elapsed = time.perf_counter() - point_start
        extra_delay = point_elapsed * (slowdown_factor - 1.0)
        time.sleep(extra_delay)
    return loss_value


def evaluate_points_on_device(
    context,
    points: Sequence[GridPoint],
    base_vector_device: torch.Tensor,
    direction_a_device: torch.Tensor,
    direction_b_device: torch.Tensor,
) -> list[tuple[int, int, float]]:
    if not points:
        return []

    if (
        context.device.type != "cpu"
        and context.compiled_gpu_chunk_eval_enabled
        and context.compiled_chunk_evaluator is not None
    ):
        records: list[tuple[int, int, float]] = []
        chunk_size = max(1, context.compiled_gpu_chunk_size)
        start = 0

        while start < len(points):
            subchunk = list(points[start : start + chunk_size])
            actual_count = len(subchunk)

            try:
                perturbations = torch.zeros(
                    (chunk_size, 2),
                    device=context.device,
                    dtype=base_vector_device.dtype,
                )
                for index in range(chunk_size):
                    point = subchunk[min(index, actual_count - 1)]
                    perturbations[index, 0] = point.alpha
                    perturbations[index, 1] = point.beta

                loss_values = evaluate_loss_compiled_chunk(
                    compiled_chunk_evaluator=context.compiled_chunk_evaluator,
                    data_loader=context.data_loader,
                    device=context.device,
                    num_batches=context.config.runtime.num_batches,
                    perturbations=perturbations,
                    active_count=actual_count,
                )
                records.extend(
                    (point.row, point.col, loss_value)
                    for point, loss_value in zip(subchunk, loss_values)
                )
                start += actual_count
            except Exception as error:
                context.compiled_chunk_evaluator = None
                print(
                    "[compile_gpu_chunk_eval] "
                    f"disabled device={context.device.type} "
                    f"reason={error}"
                )
                break

        if start == len(points):
            return records

        points = points[start:]

    return [
        (
            point.row,
            point.col,
            evaluate_point_on_device(
                context=context,
                alpha=point.alpha,
                beta=point.beta,
                base_vector_device=base_vector_device,
                direction_a_device=direction_a_device,
                direction_b_device=direction_b_device,
            ),
        )
        for point in points
    ]
