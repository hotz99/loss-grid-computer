from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from src.schemas import MLTaskSpec


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        outputs = outputs + residual

        outputs = self.relu(outputs)
        return outputs


class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 16
        self.conv1 = _conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, blocks=3, stride=1)
        self.layer2 = self._make_layer(32, blocks=3, stride=2)
        self.layer3 = self._make_layer(64, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride=stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        outputs = self.fc(outputs)
        return outputs


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


def build_model(config: MLTaskSpec) -> nn.Module:
    model = ResNet20()
    if config.checkpoint_path:
        _load_checkpoint(model, config.checkpoint_path)
    return model.float()
