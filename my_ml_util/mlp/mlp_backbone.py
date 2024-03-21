from dataclasses import dataclass

import numpy as np  # noqa
import torch
import torch.nn as nn


@dataclass
class MLPBackboneConfig:
    out_features_per_layer: list[int]
    dropout: float | None
    apply_batch_normalization: bool = False
    apply_layer_normalization: bool = False


BackboneConfig = MLPBackboneConfig  # alias, remove if not needed in the future


class Block(nn.Module):
    """The main building block of `MLP`."""

    def __init__(
            self,
            *,
            in_features: int,
            out_features: int,
            bias: bool,
            dropout: float | None,
            apply_batch_normalization: bool,
            apply_layer_normalization: bool,
    ) -> None:
        super().__init__()
        # default initialization for linear is Kaiming (aka He) uniform
        self.linear = nn.Linear(in_features, out_features, bias)
        self.batch_norm = nn.BatchNorm1d(out_features) if apply_batch_normalization else None
        self.layer_norm = nn.LayerNorm(out_features) if apply_layer_normalization else None
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0.0 else None

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.linear(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        x = self.activation(x)

        if self.dropout:
            x = self.dropout(x)
        return x


class MLPBackbone(nn.Module):
    """
    A multi-layer perceptron (MLP) backbone consisting of 1..n FC layers.
    """
    def __init__(self,
                 in_features: int,
                 out_features_per_layer: list[int],
                 dropout: float | None,
                 apply_batch_normalization: bool,
                 apply_layer_normalization: bool,):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                Block(
                    in_features=(out_features_per_layer[i - 1] if i
                                 else in_features),
                    out_features=out_features,
                    bias=True,
                    dropout=dropout,
                    apply_batch_normalization=apply_batch_normalization,
                    apply_layer_normalization=apply_layer_normalization,
                )
                for i, out_features in enumerate(out_features_per_layer)
            ]
        )

    @property
    def out_features(self) -> int:
        return self.blocks[-1].linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
