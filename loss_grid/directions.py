from __future__ import annotations

from typing import List, Tuple

import torch
from torch.nn.utils import parameters_to_vector


def _normalize_filterwise(parameter: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    if parameter.ndim <= 1:
        param_norm = torch.linalg.vector_norm(parameter.reshape(-1))
        dir_norm = torch.linalg.vector_norm(direction.reshape(-1))
        if float(dir_norm) == 0.0:
            return direction
        scale = (param_norm / dir_norm) if float(param_norm) > 0.0 else torch.tensor(1.0)
        return direction * scale.to(direction.dtype)

    flattened_param = parameter.reshape(parameter.shape[0], -1)
    flattened_dir = direction.reshape(direction.shape[0], -1)
    param_norms = torch.linalg.vector_norm(flattened_param, dim=1, keepdim=True)
    dir_norms = torch.linalg.vector_norm(flattened_dir, dim=1, keepdim=True)
    dir_norms = torch.where(dir_norms == 0, torch.ones_like(dir_norms), dir_norms)
    scales = torch.where(param_norms > 0, param_norms / dir_norms, torch.ones_like(param_norms))
    return (flattened_dir * scales).reshape_as(parameter)


def build_direction_vectors(model: torch.nn.Module, seed: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    params = [param.detach().cpu().clone() for param in model.parameters()]
    directions_a = []
    directions_b = []

    for parameter in params:
        rand_a = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
        rand_b = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
        directions_a.append(_normalize_filterwise(parameter, rand_a))
        directions_b.append(_normalize_filterwise(parameter, rand_b))

    base = parameters_to_vector(params).detach().cpu()
    vec_a = parameters_to_vector(directions_a).detach().cpu()
    vec_b = parameters_to_vector(directions_b).detach().cpu()
    return base, vec_a, vec_b
