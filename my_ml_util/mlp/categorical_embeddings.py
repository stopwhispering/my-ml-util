from dataclasses import dataclass
from typing import Callable, Literal
import math

import numpy as np  # noqa
import torch
import torch.nn as nn

ModuleType = str | Callable[..., nn.Module]
NumericalEmbedArchitectureType = Literal['MLP', 'MLP-L', 'MLP-LR', 'MLP-P', 'MLP-PL', 'MLP-PLR']


@dataclass
class CategoricalEmbeddingsConfig:
    column_idxs: tuple[int, ...]
    # number of unique values per categorical feature
    cat_column_n_unique_values: tuple[int, ...] | None = None
    embedding_max_dim: int = 64  # max size of each embedding vector
    embedding_min_dim: int = 1  # min size of each embedding vector


class CategoricalEmbeddings(nn.Module):
    def __init__(
            self,
            # n_features: int,  # number of embeddings (i.e. number of unique categories)
            # use_bias: bool,  # if True, use bias in the embedding layer
            cat_column_n_unique_values: tuple[int, ...],
            embedding_max_dim: int,  # max size of each embedding vector
            embedding_min_dim: int,  # min size of each embedding vector
            # number of unique values per categorical feature
    ) -> None:
        super().__init__()
        # if use_bias:
        #     raise NotImplementedError('use_bias is not yet implemented')
        # self.use_bias = use_bias

        self.embeddings = []
        # self.biases = []
        for i, n_unique_values in enumerate(cat_column_n_unique_values):
            # example sizes: 30 unique_values -> 10 embedding_dim
            #                 3 unique values -> 1 embedding_dim
            #                 2 unique values -> 1 embedding_dim
            #               200 unique values -> 50 embedding_dim (max is lower than 67)
            embedding_dim = min([(n_unique_values + 2) // 3, embedding_max_dim])
            embedding_dim = max([embedding_dim, embedding_min_dim])
            embed = nn.Embedding(num_embeddings=n_unique_values,
                                 embedding_dim=embedding_dim)
            setattr(self, f'emb_{i}', embed)   # req. to have it moved to correct device
            nn.init.kaiming_uniform_(embed.weight,
                                     a=math.sqrt(5))
            self.embeddings.append(embed)

    @property
    def output_dim(self) -> int:
        """Return the output dimension of the embeddings layer (per feature)."""
        # return self.embeddings.embedding_dim
        return sum(e.embedding_dim for e in self.embeddings)

    def forward(self,
                x_categorical: torch.Tensor  # [batch_size, n_cat]
                ) -> torch.Tensor:
        x_categorical = x_categorical.int()

        assert len(self.embeddings) == x_categorical.shape[1], (
            'The number of embeddings must be equal to the number of categorical columns.'
        )
        embedded_list = []
        for i in range(len(self.embeddings)):
            embedded = self.embeddings[i](x_categorical[:, i])
            # print(embedded.shape)
            embedded_list.append(embedded)

        try:
            embedded = torch.cat(embedded_list, dim=1)  # [batch_size, total_embeddings_dim]
        except RuntimeError:
            print(len(embedded_list))
            print(len(self.embeddings))
            print(self.embeddings[0].weight.shape)
            print(embedded_list[0].shape)
            raise
        return embedded
