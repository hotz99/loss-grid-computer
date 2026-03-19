from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ShuffleNet_V2_X1_0_Weights,
    mobilenet_v3_small,
    shufflenet_v2_x1_0,
)

from loss_grid.config import ModelConfig


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, use_skip: bool = True
    ):
        super().__init__()
        self.use_skip = use_skip
        self.conv1 = _conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if use_skip and (stride != 1 or in_planes != planes):
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

        if self.use_skip:
            if self.downsample is not None:
                residual = self.downsample(inputs)
            outputs = outputs + residual

        outputs = self.relu(outputs)
        return outputs


class ResNetCifar(nn.Module):
    def __init__(self, config: ModelConfig, use_skip: bool):
        super().__init__()
        if config.in_channels != 3:
            raise ValueError(
                f"ResNet20 expects 3 input channels, got {config.in_channels}"
            )

        self.in_planes = 16
        self.use_skip = use_skip
        self.conv1 = _conv3x3(config.in_channels, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, blocks=3, stride=1)
        self.layer2 = self._make_layer(32, blocks=3, stride=2)
        self.layer3 = self._make_layer(64, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, config.num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [
            BasicBlock(self.in_planes, planes, stride=stride, use_skip=self.use_skip)
        ]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(self.in_planes, planes, stride=1, use_skip=self.use_skip)
            )
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
    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}")
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key.removeprefix("module.")] = value
    model.load_state_dict(cleaned, strict=True)


def _build_torchvision_model(config: ModelConfig) -> nn.Module:
    name = config.name.lower()
    if name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if config.pretrained else None
        model = mobilenet_v3_small(weights=weights)
        if config.num_classes != 1000:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, config.num_classes)
        return model

    if name == "shufflenet_v2_x1_0":
        weights = (
            ShuffleNet_V2_X1_0_Weights.DEFAULT if config.pretrained else None
        )
        model = shufflenet_v2_x1_0(weights=weights)
        if config.num_classes != 1000:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, config.num_classes)
        return model

    raise ValueError(f"Unsupported torchvision model: {config.name}")


def build_model(config: ModelConfig) -> nn.Module:
    name = config.name.lower()
    if name == "resnet20":
        model = ResNetCifar(config, use_skip=True)
    elif name == "resnet20_no_skip":
        model = ResNetCifar(config, use_skip=False)
    elif name in {"mobilenet_v3_small", "shufflenet_v2_x1_0"}:
        return _build_torchvision_model(config)
    else:
        raise ValueError(f"Unsupported model: {config.name}")

    # torchvision models do not consume the project-specific ResNet checkpoints
    # from the base configs.
    if config.checkpoint_path and name not in {
        "mobilenet_v3_small",
        "shufflenet_v2_x1_0",
    }:
        _load_checkpoint(model, config.checkpoint_path)
    return model
