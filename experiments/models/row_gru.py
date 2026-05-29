from __future__ import annotations

import math

import torch
from torch import nn


class _GRUWeights(nn.Module):
    """GRU gate weights held as plain parameters.

    nn.GRU trips Dynamo's RNN guard during torch.compile even when it is only a
    weight container, which is the sole role it plays here. This holder keeps the
    nn.GRU parameter names, shapes, stacked gate layout, and uniform init, so
    checkpoints load unchanged and the manual recurrence stays numerically
    identical, while torch.compile can trace it.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 3 * hidden_size
        self.weight_ih_l0 = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(gate_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(gate_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -bound, bound)


class RowGRUClassifier(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.gru = _GRUWeights(input_size=96, hidden_size=hidden_size)
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
            input_gates = torch.nn.functional.linear(rows[:, row_index, :], weight_ih, bias_ih)
            hidden_gates = torch.nn.functional.linear(hidden, weight_hh, bias_hh)
            input_reset, input_update, input_new = input_gates.chunk(3, dim=1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=1)
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            new = torch.tanh(input_new + reset * hidden_new)
            hidden = (1.0 - update) * new + update * hidden

        return hidden
