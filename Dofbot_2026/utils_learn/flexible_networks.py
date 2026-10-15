"""用于学习 Dofbot 正逆运动学的小型残差多层感知机。"""

from __future__ import annotations

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.activation = nn.SiLU()

    def forward(self, inputs):
        return self.activation(inputs + self.layers(inputs))


class FlexibleMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers=(128, 128, 64),
        dropout=0.0,
        residual=True,
    ):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_layers:
            layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.SiLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            if residual:
                layers.append(ResidualBlock(width, dropout))
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)
