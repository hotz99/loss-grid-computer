from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


NamedTensorDict = OrderedDict[str, torch.Tensor]


@dataclass(frozen=True)
class ParameterLayoutEntry:
    name: str
    offset: int
    numel: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ParameterLayout:
    entries: tuple[ParameterLayoutEntry, ...]
    total_numel: int


def make_functional_state(
    module: torch.nn.Module,
) -> tuple[NamedTensorDict, NamedTensorDict, ParameterLayout]:
    named_parameters = OrderedDict(
        (name, parameter.detach()) for name, parameter in module.named_parameters()
    )
    named_buffers = OrderedDict(
        (name, buffer.detach()) for name, buffer in module.named_buffers()
    )

    entries: list[ParameterLayoutEntry] = []
    offset = 0
    for name, parameter in named_parameters.items():
        numel = int(parameter.numel())
        entries.append(
            ParameterLayoutEntry(
                name=name, offset=offset, numel=numel, shape=tuple(parameter.shape)
            )
        )
        offset += numel

    layout = ParameterLayout(entries=tuple(entries), total_numel=offset)
    return named_parameters, named_buffers, layout


def flat_chunk_to_batched_param_dict(
    flat_vectors: torch.Tensor,
    layout: ParameterLayout,
) -> NamedTensorDict:
    if flat_vectors.ndim != 2:
        raise ValueError(
            "expected a 2D chunk of flat parameter vectors, "
            f"got shape {tuple(flat_vectors.shape)}"
        )
    if int(flat_vectors.shape[1]) != layout.total_numel:
        raise ValueError(
            "flat parameter vector length does not match layout: "
            f"expected {layout.total_numel}, got {int(flat_vectors.shape[1])}"
        )

    chunk_size = int(flat_vectors.shape[0])
    return OrderedDict(
        (
            entry.name,
            flat_vectors[:, entry.offset : entry.offset + entry.numel]
            .reshape(chunk_size, *entry.shape)
            .contiguous(),
        )
        for entry in layout.entries
    )


def materialize_flat_chunk(
    chunk,
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    alphas = torch.tensor(
        [point.alpha for point in chunk], device=device, dtype=base_vector.dtype
    )
    betas = torch.tensor(
        [point.beta for point in chunk], device=device, dtype=base_vector.dtype
    )
    return (
        base_vector.unsqueeze(0)
        + alphas.unsqueeze(1) * direction_a.unsqueeze(0)
        + betas.unsqueeze(1) * direction_b.unsqueeze(0)
    )


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cuda error: out of memory" in message
        or "cudnn_status_alloc_failed" in message
        or "mps backend out of memory" in message
        or "not enough memory" in message
    )
