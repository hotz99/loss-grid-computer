from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from src.schemas import MLTaskSpec


class RowGRUClassifier(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size=96, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        rows = inputs.permute(0, 2, 1, 3).reshape(batch_size, 32, 96)
        hidden = self._manual_gru(rows)
        return self.classifier(hidden)

    def _manual_gru(self, rows: torch.Tensor) -> torch.Tensor:
        hidden_size = self.gru.hidden_size
        hidden = rows.new_zeros(rows.shape[0], hidden_size)
        weight_ih = self.gru.weight_ih_l0
        weight_hh = self.gru.weight_hh_l0
        bias_ih = self.gru.bias_ih_l0
        bias_hh = self.gru.bias_hh_l0

        for row_index in range(rows.shape[1]):
            input_gates = torch.nn.functional.linear(
                rows[:, row_index, :],
                weight_ih,
                bias_ih,
            )
            hidden_gates = torch.nn.functional.linear(
                hidden,
                weight_hh,
                bias_hh,
            )
            input_reset, input_update, input_new = input_gates.chunk(3, dim=1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=1)
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            new = torch.tanh(input_new + reset * hidden_new)
            hidden = (1.0 - update) * new + update * hidden

        return hidden


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
    model = RowGRUClassifier()
    if spec.checkpoint_path:
        _load_checkpoint(model, spec.checkpoint_path)
    return model.float()
