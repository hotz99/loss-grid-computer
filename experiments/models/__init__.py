from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from experiments.models.mlp_regressor import MLPRegressor
from experiments.models.mnist_mlp import MnistMLP
from experiments.models.resnet20 import ResNet20
from experiments.models.row_gru import RowGRUClassifier
from experiments.paths import resolve_asset
from experiments.schemas import MLTaskSpec


__all__ = ["build_model", "load_checkpoint"]


def build_model(task: MLTaskSpec) -> nn.Module:
    model = _construct(task)
    if task.checkpoint_path:
        load_checkpoint(model, task.checkpoint_path)
    return model.float()


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    path = resolve_asset(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}")
    cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if torch.is_floating_point(value):
            value = value.to(torch.float32)
        cleaned[key.removeprefix("module.")] = value
    model.load_state_dict(cleaned, strict=True)


def _construct(task: MLTaskSpec) -> nn.Module:
    if task.model == "resnet20":
        return ResNet20()
    if task.model == "row_gru":
        return RowGRUClassifier()
    if task.model == "mlp_regressor":
        return MLPRegressor(task.dataset.input_shape[0])
    if task.model == "mlp" or task.model == "mnist_mlp":
        return MnistMLP()
    raise ValueError(f"unsupported model: {task.model}")
