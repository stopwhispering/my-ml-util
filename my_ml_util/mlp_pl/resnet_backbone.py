from dataclasses import dataclass

import pytorch_lightning as pl
import numpy as np  # noqa
import torch
import torch.nn as nn


@dataclass
class ResnetBackboneConfig:
    hidden_layer_size_in: int
    hidden_layer_size_out: int
    n_residual_blocks: int
    dropout: float | None
    apply_batch_normalization: bool = False


class ResnetBlock(pl.LightningModule):
    """The main building block of ResNet. Consists of two linear layers.."""

    def __init__(
            self,
            *,
            hidden_layer_size_in: int,
            hidden_layer_size_out: int,
            dropout: float | None,
            apply_batch_normalization: bool,
    ) -> None:
        super().__init__()
        # default initialization for linear is Kaiming (aka He) uniform
        self.linear_0 = nn.Linear(hidden_layer_size_in, hidden_layer_size_out)
        self.linear_1 = nn.Linear(hidden_layer_size_out, hidden_layer_size_in)
        self.batch_norm = nn.BatchNorm1d(hidden_layer_size_in) if apply_batch_normalization else None
        self.activation = nn.ReLU()
        self.dropout = (nn.Dropout(dropout)
                        if dropout is not None and dropout > 0.0 else None)

    def forward(self,
                x: torch.Tensor  # [batch_size, hidden_layer_size_in] not activated, not normalized
                ) -> torch.Tensor:
        z = x
        if self.batch_norm:
            z = self.batch_norm(z)
        z = self.linear_0(z)  # [batch_size, hidden_layer_size_out]
        z = self.activation(z)
        if self.dropout:
            z = self.dropout(z)
        z = self.linear_1(z)  # [batch_size, hidden_layer_size_in]
        # if self.dropout:  # we could add another dropout here with another p value
        #     z = self.dropout(z)

        # now comes the residual connection
        x = x + z  # that's the input to the next residual block (or the head)
        return x  # [batch_size, hidden_layer_size_in]


class ResnetBackbone(pl.LightningModule):
    """
    A ResNet backbone (residual connections aka skip connections) for MLP.
    """
    def __init__(self,
                 in_features: int,
                 hidden_layer_size_in: int,
                 hidden_layer_size_out: int,
                 n_residual_blocks: int,  # each block consists of two linear layers
                 dropout: float | None,
                 apply_batch_normalization: bool,):
        super().__init__()

        self.hidden_layer_size_in = hidden_layer_size_in

        self.first_layer = nn.Linear(in_features, hidden_layer_size_in)
        self.blocks = nn.Sequential(
            *[
                ResnetBlock(
                    hidden_layer_size_in=hidden_layer_size_in,
                    hidden_layer_size_out=hidden_layer_size_out,
                    dropout=dropout,
                    apply_batch_normalization=apply_batch_normalization,
                )
                for _ in range(n_residual_blocks)
            ]
        )
        self.batch_norm = nn.BatchNorm1d(hidden_layer_size_in) if apply_batch_normalization else None
        self.activation = nn.ReLU()

    def forward(self,
                x: torch.Tensor  # [batch_size, in_features]
                ) -> torch.Tensor:
        x = self.first_layer(x)  # [batch_size, hidden_layer_size_in]
        x = self.blocks(x)  # [batch_size, hidden_layer_size_in]
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x

    @property
    def out_features(self) -> int:
        return self.hidden_layer_size_in
