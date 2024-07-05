from dataclasses import dataclass
from typing import Callable, Literal
import math

import pytorch_lightning as pl
import numpy as np  # noqa
import torch
import torch.nn as nn


ModuleType = str | Callable[..., pl.LightningModule]
NumericalEmbedArchitectureType = Literal['MLP', 'MLP-L', 'MLP-LR', 'MLP-P', 'MLP-PL', 'MLP-PLR']


@dataclass
class NumericalEmbeddingsConfig:
    architecture: NumericalEmbedArchitectureType
    column_idxs: tuple[int, ...]      # indices of the numerical columns for num embed.
    linear_embedding_dim: int = 35    # size of the linear embedding vector
    periodic_embedding_dim: int = 30  # size of the periodic embedding vector
    periodic_sigma: float = 0.10      # sigma for the periodic embeddings


class PeriodicEmbeddings(pl.LightningModule):
    """Generate embeddings for numerical features using periodic activation functions"""

    def __init__(self,
                 n_features: int,
                 embedding_dim: int,  # size of each embedding vector
                 sigma: float,  # sigma for the periodic embeddings
                 ) -> None:
        super().__init__()
        coefficients = torch.normal(mean=0.0,
                                    std=sigma,
                                    size=(n_features, embedding_dim))  # [n_features, embedding_dim]
        self.coefficients = nn.Parameter(coefficients)

    @property
    def output_dim(self) -> int:
        """Return the output dimension of the embeddings layer (per feature)."""
        return self.coefficients.shape[1] * 2

    def forward(self,
                x: torch.Tensor  # [batch_size, n_features]
                ) -> torch.Tensor:
        assert x.ndim == 2
        x = (
                2 * torch.pi * self.coefficients[None]  # [1, n_features, embedding_dim]
                * x[..., None]  # [batch_size, n_features, 1]
        )  # [batch_size, n_features, embedding_dim]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)  # [batch_size, n_features, embedding_dim*2]
        return x


class LinearEmbeddings(pl.LightningModule):
    """'Conventional' differentiable embedding module based on Linear layers."""

    def __init__(self,
                 n_features: int,  # number of features
                 embedding_dim: int,  # size of each embedding vector
                 ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_features, embedding_dim))
        self.bias = nn.Parameter(torch.Tensor(n_features, embedding_dim))

        # initialization
        embedding_dim_sqrt_inv = 1 / math.sqrt(embedding_dim)
        nn.init.uniform_(self.weight, a=-embedding_dim_sqrt_inv, b=embedding_dim_sqrt_inv)
        nn.init.uniform_(self.bias, a=-embedding_dim_sqrt_inv, b=embedding_dim_sqrt_inv)

    @property
    def output_dim(self) -> int:
        """Return the output dimension of the embeddings layer (per feature)."""
        return self.weight.shape[1]

    def forward(self,
                x: torch.Tensor  # [batch_size, n_features]
                ) -> torch.Tensor:
        assert x.ndim == 2
        x = (
                self.weight[None] *  # [1, n_features, embedding_dim]
                x[..., None]  # [batch_size, n_features, 1]
        )
        x = (
                x +  # [batch_size, n_features, embedding_dim]
                self.bias[None]  # [1, n_features, embedding_dim]
        )
        return x  # [batch_size, n_features, embedding_dim]


class SequentLinearEmbeddings(pl.LightningModule):
    """'Conventional' differentiable embedding module based on Linear layers. Implementation for usage
    at position 2+ in a sequence of layers, e.g. after Periodic Embedings."""

    def __init__(self,
                 n_features: int,  # number of features
                 input_dim: int,
                 # size of each input vector from the previous layer (e.g. PeriodicEmbeddings)
                 embedding_dim: int,  # size of each embedding vector
                 ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_features, input_dim, embedding_dim))
        self.bias = nn.Parameter(torch.Tensor(n_features, embedding_dim))

        # initialization
        embedding_dim_sqrt_inv = 1 / math.sqrt(embedding_dim)
        nn.init.uniform_(self.weight, a=-embedding_dim_sqrt_inv, b=embedding_dim_sqrt_inv)
        nn.init.uniform_(self.bias, a=-embedding_dim_sqrt_inv, b=embedding_dim_sqrt_inv)

    @property
    def output_dim(self) -> int:
        """Return the output dimension of the embeddings layer (per feature)."""
        return self.weight.shape[2]

    def forward(self,
                x: torch.Tensor  # [batch_size, n_features, output_dim_previous_layer]
                ) -> torch.Tensor:
        assert x.ndim == 3

        x = (
                self.weight[None] *  # [1, n_features, output_dim_previous_layer, embedding_dim]
                x[..., None]  # [batch_size, n_features, output_dim_previous_layer, 1]
        )  # [batch_size, n_features, output_dim_previous_layer, embedding_dim]
        x = x.sum(-2)  # [batch_size, n_features, embedding_dim]
        x = (
                x +
                self.bias[None]  # [1, n_features, embedding_dim]
        )
        return x  # [batch_size, n_features, embedding_dim]


class NumericalEmbeddings(pl.LightningModule):
    def __init__(
            self,
            n_features: int,  # number of numerical feat.
            # embedding_dim: int,  # size of each embedding vector
            numerical_embed_architecture: NumericalEmbedArchitectureType,

            linear_embedding_dim: int | None,  # size of the linear embedding vector
            periodic_embedding_dim: int,  # size of the periodic embedding vector
            periodic_sigma: float,  # sigma for the periodic embeddings
    ) -> None:
        super().__init__()

        layers = []
        if numerical_embed_architecture in {'MLP-L', 'MLP-LR'}:
            assert linear_embedding_dim is not None
            layers.append(
                LinearEmbeddings(n_features=n_features,
                                 embedding_dim=linear_embedding_dim,
                                 )
            )

            if numerical_embed_architecture == 'MLP-LR':
                layers.append(nn.ReLU())

        elif numerical_embed_architecture in {'MLP-P', 'MLP-PL', 'MLP-PLR'}:
            assert periodic_embedding_dim is not None
            periodic_embeddings = PeriodicEmbeddings(n_features=n_features,
                                                     embedding_dim=periodic_embedding_dim,
                                                     sigma=periodic_sigma,
                                                     )
            layers.append(periodic_embeddings)

            if numerical_embed_architecture in {'MLP-PL', 'MLP-PLR'}:
                layers.append(SequentLinearEmbeddings(n_features=n_features,
                                                      input_dim=periodic_embeddings.output_dim,
                                                      embedding_dim=linear_embedding_dim, ))

                if numerical_embed_architecture == 'MLP-PLR':
                    layers.append(nn.ReLU())

        else:
            raise ValueError(
                f"Unknown numerical_embed_architecture: {numerical_embed_architecture}")

        # if use_relu_embedding:
        #     layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        """Return the output dimension of the (final) embeddings layer (per feature)."""
        final_layer = self.layers[-1] if not isinstance(self.layers[-1], nn.ReLU) else self.layers[
            -2]
        return final_layer.output_dim

    def forward(self, x):
        return self.layers(x)
