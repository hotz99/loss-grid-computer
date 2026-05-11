from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from src.schemas import MLTaskSpec


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}")
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if torch.is_floating_point(value):
            value = value.to(torch.float32)
        cleaned[key.removeprefix("module.")] = value
    model.load_state_dict(cleaned, strict=True)


def build_model(spec: MLTaskSpec) -> nn.Module:
    model = MLPRegressor(spec.dataset.input_shape[0])
    if spec.checkpoint_path:
        _load_checkpoint(model, spec.checkpoint_path)
    return model.float()
