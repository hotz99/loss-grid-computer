from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from src.config import GridConfig


@dataclass(frozen=True)
class GridPoint:
    linear_idx: int
    row: int
    col: int
    alpha: float
    beta: float


def build_grid_points(config: GridConfig) -> List[GridPoint]:
    alphas = torch.linspace(-1.0, 1.0, config.resolution).tolist()
    betas = torch.linspace(-1.0, 1.0, config.resolution).tolist()
    points = []
    linear_idx = 0
    for row, alpha in enumerate(alphas):
        for col, beta in enumerate(betas):
            points.append(
                GridPoint(
                    linear_idx=linear_idx, row=row, col=col, alpha=alpha, beta=beta
                )
            )
            linear_idx += 1
    return points

def _row_partition(
    points: Sequence[GridPoint],
    resolution: int,
    worker_index: int,
    worker_count: int,
) -> List[GridPoint]:
    rows_per_worker = (resolution + worker_count - 1) // worker_count
    start_row = worker_index * rows_per_worker
    end_row = min(resolution, start_row + rows_per_worker)
    return [point for point in points if start_row <= point.row < end_row]


def partition_points(
    points: Sequence[GridPoint],
    grid_config: GridConfig,
    worker_index: int,
    worker_count: int,
) -> List[GridPoint]:
    return _row_partition(points, grid_config.resolution, worker_index, worker_count)
