from __future__ import annotations

from torch import nn

from orbit_cifar.models.resnet_gn import resnet20_gn
from orbit_cifar.models.mlp_ln import MLPLayerNorm


def create_model(
    name: str = "resnet20_gn",
    num_classes: int = 10,
    group_norm_max_groups: int = 8,
    input_channels: int = 3,
    input_height: int = 32,
    input_width: int = 32,
    hidden_dimensions: tuple[int, int] | list[int] = (256, 128),
) -> nn.Module:
    if name == "resnet20_gn":
        return resnet20_gn(num_classes=num_classes, maximum_groups=group_norm_max_groups, input_channels=input_channels)
    if name == "mlp_ln":
        return MLPLayerNorm((input_channels, input_height, input_width), num_classes, hidden_dimensions)
    if name == "tiny_cnn":
        return nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, num_classes),
        )
    raise ValueError(f"Unknown model architecture: {name}")
