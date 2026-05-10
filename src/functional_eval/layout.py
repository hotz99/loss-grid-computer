from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

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

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)


def extract_named_parameters(module: torch.nn.Module) -> NamedTensorDict:
    """Return detached parameters in PyTorch's stable module traversal order."""
    return OrderedDict(
        (name, parameter.detach())
        for name, parameter in module.named_parameters()
    )


def extract_named_buffers(module: torch.nn.Module) -> NamedTensorDict:
    """Return detached buffers in PyTorch's stable module traversal order."""
    return OrderedDict(
        (name, buffer.detach())
        for name, buffer in module.named_buffers()
    )


def build_parameter_layout(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
) -> ParameterLayout:
    entries: list[ParameterLayoutEntry] = []
    offset = 0

    for name, parameter in named_parameters:
        numel = int(parameter.numel())
        entries.append(
            ParameterLayoutEntry(
                name=name,
                offset=offset,
                numel=numel,
                shape=tuple(parameter.shape),
            )
        )
        offset += numel

    return ParameterLayout(entries=tuple(entries), total_numel=offset)


def make_functional_state(
    module: torch.nn.Module,
) -> tuple[NamedTensorDict, NamedTensorDict, ParameterLayout]:
    named_parameters = extract_named_parameters(module)
    named_buffers = extract_named_buffers(module)
    layout = build_parameter_layout(named_parameters.items())
    return named_parameters, named_buffers, layout


def flat_vector_to_param_dict(
    flat_vector: torch.Tensor,
    layout: ParameterLayout,
) -> NamedTensorDict:
    if flat_vector.ndim != 1:
        raise ValueError(
            f"expected a 1D flat parameter vector, got shape {tuple(flat_vector.shape)}"
        )
    _validate_flat_numel(flat_vector.numel(), layout)

    return OrderedDict(
        (
            entry.name,
            flat_vector.narrow(0, entry.offset, entry.numel).view(entry.shape),
        )
        for entry in layout.entries
    )


def flat_chunk_to_batched_param_dict(
    flat_vectors: torch.Tensor,
    layout: ParameterLayout,
) -> NamedTensorDict:
    if flat_vectors.ndim != 2:
        raise ValueError(
            "expected a 2D chunk of flat parameter vectors, "
            f"got shape {tuple(flat_vectors.shape)}"
        )
    _validate_flat_numel(flat_vectors.shape[1], layout)

    chunk_size = int(flat_vectors.shape[0])
    return OrderedDict(
        (
            entry.name,
            flat_vectors.as_strided(
                size=(chunk_size, *entry.shape),
                stride=(
                    flat_vectors.stride(0),
                    *(
                        stride * flat_vectors.stride(1)
                        for stride in _contiguous_strides(entry.shape)
                    ),
                ),
                storage_offset=(
                    flat_vectors.storage_offset()
                    + entry.offset * flat_vectors.stride(1)
                ),
            ),
        )
        for entry in layout.entries
    )


def _validate_flat_numel(actual_numel: int, layout: ParameterLayout) -> None:
    if int(actual_numel) != layout.total_numel:
        raise ValueError(
            "flat parameter vector length does not match layout: "
            f"expected {layout.total_numel}, got {int(actual_numel)}"
        )


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides: list[int] = []
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))
