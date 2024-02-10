"""
cf. https://github.com/yandex-research/rtdl/blob/main/rtdl/modules.py
Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
"""
import random
from typing import Callable, Literal
import math

import numpy as np  # noqa
import torch
import torch.nn as nn

ModuleType = str | Callable[..., nn.Module]
NumericalEmbedArchitectureType = Literal['MLP', 'MLP-L', 'MLP-LR', 'MLP-P', 'MLP-PL', 'MLP-PLR']


def seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False  # warning: this might impair performance


class Block(nn.Module):
    """The main building block of `MLP`."""

    def __init__(
            self,
            *,
            in_features: int,
            out_features: int,
            bias: bool,
            dropout: float | None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)

        if self.dropout:
            x = self.dropout(x)
        return x


class CategoricalEmbeddings(nn.Module):
    def __init__(
            self,
            n_features: int,  # number of embeddings (i.e. number of unique categories)
            embedding_dim,  # size of each embedding vector
            use_bias: bool,  # if True, use bias in the embedding layer
    ) -> None:
        super().__init__()
        self.use_bias = use_bias

        self.embeddings = nn.Embedding(
            num_embeddings=n_features,  # size of the dictionary of embeddings
            embedding_dim=embedding_dim,  # size of each embedding vector
        )
        nn.init.kaiming_uniform_(self.embeddings.weight,
                                 a=math.sqrt(5))
        print(f'{self.embeddings.weight.shape=}')

        if use_bias:
            self.embedding_bias = nn.Parameter(torch.Tensor(n_features,
                                                            embedding_dim))
            nn.init.kaiming_uniform_(self.embedding_bias,
                                     a=math.sqrt(5))

    @property
    def output_dim(self) -> int:
        """Return the output dimension of the embeddings layer (per feature)."""
        return self.embeddings.embedding_dim

    def forward(self,
                x_categorical: torch.Tensor  # [batch_size, n_cat]
                ) -> torch.Tensor:
        x_categorical = x_categorical.int()

        x_categorical = self.embeddings(x_categorical)  # [batch_size, n_cat, emb_dim]
        if self.use_bias:
            x_categorical = x_categorical + self.embedding_bias[None]

        # we concatenate the embeddings vectors to a 1-d tensor per sample
        x_categorical = x_categorical.view(x_categorical.size(0),
                                           -1)  # [batch_size, n_cat * emb_dim]
        return x_categorical


class PeriodicEmbeddings(nn.Module):
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


class LinearEmbeddings(nn.Module):
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


class SequentLinearEmbeddings(nn.Module):
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


class NumericalEmbeddings(nn.Module):
    def __init__(
            self,
            n_features: int,  # number of embeddings (i.e. number of numerical feat.)
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


class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].
    The following scheme describes the architecture:
    .. code-block:: text
          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)
    """

    def __init__(
            self,
            *,
            cat_column_idxs: tuple[int, ...],  # indices of the categorical columns for cat embed.
            num_column_idxs: tuple[int, ...],  # indices of the numerical columns for num embed.
            passthrough_column_idxs: tuple[int, ...],
            # indices of the columns to be passed through w/o embed

            out_features_per_layer: list[int],  # dimensions of the linear layers
            dropout: float | None,  # dropout rate for all hidden layers
            dim_out: int,  # output size

            headless=False,  # if True, head is not used in prediction (no heads in pretraining)

            use_cat_embedding: bool = False,  # if True, use categorical embeddings
            cat_embedding_dim: int = 50,  # size of each embedding vector
            cat_embedding_use_bias: bool = False,  # if True, use bias in the embedding layer

            numerical_embed_architecture: NumericalEmbedArchitectureType = 'MLP',

            # num_embedding_dim: int = 45,  # size of each embedding vector
            linear_embedding_dim: int = 35,  # size of the linear embedding vector
            periodic_embedding_dim: int = 30,  # size of the periodic embedding vector
            periodic_sigma: float = 0.10,  # sigma for the periodic embeddings
            # use_positional_embedding: bool = True,
            # if True, use positional embedding for numerical features
            # use_linear_embedding: bool = True,
            # if True, use linear embedding for numerical features
            # use_relu_embedding: bool = True,  # if True, use ReLU embedding for numerical features

    ) -> None:
        super().__init__()
        self.headless = headless
        self.use_cat_embedding = use_cat_embedding
        self.cat_column_idxs = cat_column_idxs
        # self.cat_embedding_use_bias = cat_embedding_use_bias
        self.num_column_idxs = num_column_idxs
        self.passthrough_column_idxs = passthrough_column_idxs
        self.numerical_embed_architecture = numerical_embed_architecture

        assert not (use_cat_embedding and not cat_column_idxs), (
            'If use_categorical_embedding is True, then categorical_column_idxs must be provided.'
        )

        if len(out_features_per_layer) > 2:
            assert len(set(out_features_per_layer[1:-1])) == 1, (
                'If out_features_per_layer contains more than three elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )

        if use_cat_embedding:
            self.cat_embeddings = CategoricalEmbeddings(
                n_features=len(cat_column_idxs),
                embedding_dim=cat_embedding_dim,
                use_bias=cat_embedding_use_bias,
            )
        else:
            self.cat_embeddings = None

        self.num_embeddings = NumericalEmbeddings(
            n_features=len(num_column_idxs),
            # embedding_dim=num_embedding_dim,
            numerical_embed_architecture=numerical_embed_architecture,
            linear_embedding_dim=linear_embedding_dim,
            periodic_embedding_dim=periodic_embedding_dim,
            periodic_sigma=periodic_sigma,
        ) if self.numerical_embed_architecture != 'MLP' else None

        self.backbone = nn.Sequential(
            *[
                Block(
                    in_features=out_features_per_layer[i - 1] if i else self.backbone_in_features,
                    out_features=out_features,
                    bias=True,
                    dropout=dropout,
                )
                for i, out_features in enumerate(out_features_per_layer)
            ]
        )

        if not headless:
            self.head = nn.Linear(out_features_per_layer[-1], dim_out)

    @property
    def backbone_in_features(self) -> int:
        """Return the number of input features for the first mlp backbone layer."""
        if self.cat_embeddings:
            cat_dim = len(self.cat_column_idxs) * self.cat_embeddings.output_dim
        else:
            cat_dim = len(self.cat_column_idxs)

        if self.num_embeddings:
            num_dim = len(self.num_column_idxs) * self.num_embeddings.output_dim
        else:
            num_dim = len(self.num_column_idxs)

        return cat_dim + num_dim + len(self.passthrough_column_idxs)

    def forward(self,
                x: torch.Tensor  #
                ) -> torch.Tensor:
        if self.cat_embeddings or self.num_embeddings:

            # assert x.shape[1] == (len(self.cat_column_idxs)
            #                       + len(self.num_column_idxs)
            #                       + len(self.passthrough_column_idxs)), (
            #     'The input tensor must have the same number of columns as the sum of the categorical'
            #     ' and numerical column indices plus passthrough columns.'
            # )
            x_categorical = x[:, self.cat_column_idxs]
            x_numerical = x[:, self.num_column_idxs]
            X_passthrough = x[:, self.passthrough_column_idxs]

            if self.cat_embeddings:
                x_categorical = self.cat_embeddings(x_categorical)  # [batch_size, n_cat, emb_dim]

            if self.num_embeddings:
                x_numerical = self.num_embeddings(x_numerical)  # [batch_size, n_num, final_emb_dim]
                x_numerical = x_numerical.flatten(1, 2)  # [batch_size, n_num * final_emb_dim]

            x = torch.cat([x_numerical, x_categorical, X_passthrough],
                          dim=1)  # [batch_size, n_num + n_cat * emb_dim]

        else:
            x = x[:, self.x_categorical_column_idxs +
                     self.x_numerical_column_idxs +
                     self.x_passthrough_column_idxs]

        x = self.backbone(x)
        if not self.headless:
            x = self.head(x)
        return x
