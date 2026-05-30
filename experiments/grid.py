from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch.nn.utils import parameters_to_vector

from experiments.schemas import GridSpec


@dataclass(frozen=True, slots=True)
class GridPoint:
    linear_idx: int
    row: int
    col: int
    alpha: float
    beta: float


def build_grid_points(grid: GridSpec) -> List[GridPoint]:
    alphas = torch.linspace(-grid.scale, grid.scale, grid.resolution).tolist()
    betas = torch.linspace(-grid.scale, grid.scale, grid.resolution).tolist()
    points: List[GridPoint] = []
    linear_idx = 0
    for row, alpha in enumerate(alphas):
        for col, beta in enumerate(betas):
            points.append(GridPoint(linear_idx, row, col, alpha, beta))
            linear_idx += 1
    return points


def build_direction_vectors(
    model: torch.nn.Module,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Li et al. 2D filter-normalized directions [li2018losslandscape]."""
    generator = torch.Generator().manual_seed(seed)
    params = [
        parameter.detach().cpu().to(torch.float32).clone()
        for parameter in model.parameters()
    ]
    directions_a: list[torch.Tensor] = []
    directions_b: list[torch.Tensor] = []

    for parameter in params:
        rand_a = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
        rand_b = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
        directions_a.append(_normalize_filterwise(parameter, rand_a))
        directions_b.append(_normalize_filterwise(parameter, rand_b))

    base = parameters_to_vector(params).detach().cpu()
    vec_a = parameters_to_vector(directions_a).detach().cpu().to(torch.float32)
    vec_b = parameters_to_vector(directions_b).detach().cpu().to(torch.float32)
    return base, vec_a, vec_b


def perturb(
    base: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    return base + (alpha * direction_a) + (beta * direction_b)


def _normalize_filterwise(
    parameter: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    # Li et al. ignore='biasbn': zero the direction for 1D parameters (biases
    # and BatchNorm affine weights), perturbing only conv/linear filters.
    if parameter.ndim <= 1:
        return torch.zeros_like(direction)

    flattened_param = parameter.reshape(parameter.shape[0], -1)
    flattened_dir = direction.reshape(direction.shape[0], -1)
    param_norms = torch.linalg.vector_norm(flattened_param, dim=1, keepdim=True)
    dir_norms = torch.linalg.vector_norm(flattened_dir, dim=1, keepdim=True)
    dir_norms = torch.where(dir_norms == 0, torch.ones_like(dir_norms), dir_norms)
    scales = torch.where(
        param_norms > 0, param_norms / dir_norms, torch.ones_like(param_norms)
    )
    return (flattened_dir * scales).reshape_as(parameter)
