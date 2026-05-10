from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from src.system_schema import MLTaskSpec


class RowGRUClassifier(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size=96, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        rows = inputs.permute(0, 2, 1, 3).reshape(batch_size, 32, 96)
        _, hidden = self.gru(rows)
        return self.classifier(hidden[-1])


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}")
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if torch.is_floating_point(value):
            value = value.to(torch.float32)
        cleaned[key.removeprefix("module.")] = value
    model.load_state_dict(cleaned, strict=True)


def build_model(spec: MLTaskSpec) -> nn.Module:
    model = RowGRUClassifier()
    if spec.checkpoint_path:
        _load_checkpoint(model, spec.checkpoint_path)
    return model.float()
