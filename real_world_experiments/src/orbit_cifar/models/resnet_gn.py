"""CIFAR-style ResNet-20 using GroupNorm exclusively."""

from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int, maximum: int) -> int:
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    raise ValueError(f"Could not find a valid GroupNorm group count for {channels} channels")


def _normalization(channels: int, maximum_groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(_group_count(channels, maximum_groups), channels)


class BasicBlock(nn.Module):
    """Two-convolution residual block with a projection when dimensions change."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, maximum_groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _normalization(out_channels, maximum_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _normalization(out_channels, maximum_groups)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _normalization(out_channels, maximum_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.relu(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return self.relu(outputs + residual)


class ResNetCIFAR(nn.Module):
    """ResNet-20 topology adapted to 32x32 images without max pooling."""

    def __init__(self, num_classes: int = 10, maximum_groups: int = 8, input_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = 16
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1, bias=False),
            _normalization(16, maximum_groups),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(16, blocks=3, stride=1, maximum_groups=maximum_groups)
        self.stage2 = self._make_stage(32, blocks=3, stride=2, maximum_groups=maximum_groups)
        self.stage3 = self._make_stage(64, blocks=3, stride=2, maximum_groups=maximum_groups)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
        self.reset_parameters()

    def _make_stage(self, channels: int, blocks: int, stride: int, maximum_groups: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, channels, stride, maximum_groups)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels, 1, maximum_groups) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def reset_parameters(self) -> None:
        """Apply Kaiming initialization and standard affine defaults."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.pool(outputs).flatten(1)
        return self.classifier(outputs)


def resnet20_gn(num_classes: int = 10, maximum_groups: int = 8, input_channels: int = 3) -> ResNetCIFAR:
    return ResNetCIFAR(num_classes=num_classes, maximum_groups=maximum_groups, input_channels=input_channels)
