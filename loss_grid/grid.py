from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from loss_grid.config import DecompositionConfig, GridConfig


@dataclass(frozen=True)
class GridPoint:
    linear_idx: int
    row: int
    col: int
    alpha: float
    beta: float


def build_grid_points(config: GridConfig) -> List[GridPoint]:
    alphas = torch.linspace(config.alpha_min, config.alpha_max, config.resolution).tolist()
    betas = torch.linspace(config.beta_min, config.beta_max, config.resolution).tolist()
    points = []
    linear_idx = 0
    for row, alpha in enumerate(alphas):
        for col, beta in enumerate(betas):
            points.append(GridPoint(linear_idx=linear_idx, row=row, col=col, alpha=alpha, beta=beta))
            linear_idx += 1
    return points


def _row_partition(points: Sequence[GridPoint], resolution: int, rank: int, workers: int) -> List[GridPoint]:
    rows_per_worker = (resolution + workers - 1) // workers
    start_row = rank * rows_per_worker
    end_row = min(resolution, start_row + rows_per_worker)
    return [point for point in points if start_row <= point.row < end_row]


def _block_partition(
    points: Sequence[GridPoint],
    resolution: int,
    rank: int,
    workers: int,
    config: DecompositionConfig,
) -> List[GridPoint]:
    tile_h = max(1, config.tile_rows)
    tile_w = max(1, config.tile_cols)
    local = []
    for point in points:
        tile_row = point.row // tile_h
        tile_col = point.col // tile_w
        tiles_per_row = (resolution + tile_w - 1) // tile_w
        owner = (tile_row * tiles_per_row + tile_col) % workers
        if owner == rank:
            local.append(point)
    return local


def _cyclic_partition(points: Sequence[GridPoint], rank: int, workers: int) -> List[GridPoint]:
    return [point for point in points if point.linear_idx % workers == rank]


def partition_points(
    points: Sequence[GridPoint],
    grid_config: GridConfig,
    decomposition: DecompositionConfig,
    rank: int,
    workers: int,
) -> List[GridPoint]:
    strategy = decomposition.strategy.lower()
    if strategy == "row":
        return _row_partition(points, grid_config.resolution, rank, workers)
    if strategy == "block":
        return _block_partition(points, grid_config.resolution, rank, workers, decomposition)
    if strategy == "cyclic":
        return _cyclic_partition(points, rank, workers)
    raise ValueError(f"Unsupported decomposition strategy: {decomposition.strategy}")
