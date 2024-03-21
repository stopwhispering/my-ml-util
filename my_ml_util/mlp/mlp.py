"""
cf. https://github.com/yandex-research/rtdl/blob/main/rtdl/modules.py
Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
"""
import random
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np  # noqa
import torch
import torch.nn as nn

from my_ml_util.mlp.mlp_backbone import MLPBackboneConfig, MLPBackbone
from my_ml_util.mlp.categorical_embeddings import CategoricalEmbeddings, CategoricalEmbeddingsConfig
from my_ml_util.mlp.numerical_embeddings import NumericalEmbeddings, NumericalEmbeddingsConfig
from my_ml_util.mlp.resnet_backbone import ResnetBackboneConfig, ResnetBackbone

ModuleType = str | Callable[..., nn.Module]
NumericalEmbedArchitectureType = Literal['MLP', 'MLP-L', 'MLP-LR', 'MLP-P', 'MLP-PL', 'MLP-PLR']


def seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # warning: this might impair performance


@dataclass
class HeadConfig:
    dim_out: int


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
            n_features: int,
            backbone_config: MLPBackboneConfig | ResnetBackboneConfig,
            head_config: HeadConfig,
            numerical_embeddings_config: NumericalEmbeddingsConfig | None = None,
            categorical_embeddings_config: CategoricalEmbeddingsConfig | None = None,

            headless=False,  # if True, head is not used in prediction (no heads in pretraining)
    ) -> None:
        super().__init__()
        self.headless = headless
        self.cat_column_idxs: tuple[int, ...]
        self.num_column_idxs: tuple[int, ...]
        self.passthrough_column_idxs: tuple[int, ...]
        (self.cat_column_idxs,
         self.num_column_idxs,
         self.passthrough_column_idxs) = self.get_column_indexes(numerical_embeddings_config,
                                                                 categorical_embeddings_config,
                                                                 n_features)

        if categorical_embeddings_config is not None and len(
                categorical_embeddings_config.column_idxs) > 0:
            assert (categorical_embeddings_config.cat_column_n_unique_values is not None and
                    len(categorical_embeddings_config.cat_column_n_unique_values) ==
                    len(categorical_embeddings_config.column_idxs)), (
                'The number of unique values per categorical feature must be specified for each'
                ' categorical feature.'
            )

            self.cat_embeddings = CategoricalEmbeddings(
                cat_column_n_unique_values=categorical_embeddings_config.cat_column_n_unique_values,
                embedding_min_dim=categorical_embeddings_config.embedding_min_dim,
                embedding_max_dim=categorical_embeddings_config.embedding_max_dim,
            )
        else:
            self.cat_embeddings = None

        self.num_embeddings = NumericalEmbeddings(
            n_features=len(numerical_embeddings_config.column_idxs),
            numerical_embed_architecture=numerical_embeddings_config.architecture,
            linear_embedding_dim=numerical_embeddings_config.linear_embedding_dim,
            periodic_embedding_dim=numerical_embeddings_config.periodic_embedding_dim,
            periodic_sigma=numerical_embeddings_config.periodic_sigma,
        ) if (numerical_embeddings_config is not None and
              numerical_embeddings_config.architecture != 'MLP') else None

        if isinstance(backbone_config, ResnetBackboneConfig):
            self.backbone = ResnetBackbone(
                in_features=self.backbone_in_features,
                hidden_layer_size_in=backbone_config.hidden_layer_size_in,
                hidden_layer_size_out=backbone_config.hidden_layer_size_out,
                n_residual_blocks=backbone_config.n_residual_blocks,
                dropout=backbone_config.dropout,
                apply_batch_normalization=backbone_config.apply_batch_normalization,
            )

        elif isinstance(backbone_config, MLPBackboneConfig):
            self.backbone = MLPBackbone(
                in_features=self.backbone_in_features,
                out_features_per_layer=backbone_config.out_features_per_layer,
                dropout=backbone_config.dropout,
                apply_batch_normalization=backbone_config.apply_batch_normalization,
                apply_layer_normalization=backbone_config.apply_layer_normalization,
            )
        else:
            raise ValueError(
                f'backbone_config must be either MLPBackboneConfig or ResnetBackboneConfig,'
                f' not {type(backbone_config)}')

        if not headless:
            # self.head = nn.Linear(backbone_config.out_features_per_layer[-1], head_config.dim_out)
            self.head = nn.Linear(self.backbone.out_features,
                                  head_config.dim_out)

    @staticmethod
    def get_column_indexes(numerical_embeddings_config: NumericalEmbeddingsConfig | None,
                           categorical_embeddings_config: CategoricalEmbeddingsConfig | None,
                           n_features: int,
                           ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        cat_column_idxs = (categorical_embeddings_config.column_idxs
                           if categorical_embeddings_config is not None else ())
        num_column_idxs = (numerical_embeddings_config.column_idxs
                           if numerical_embeddings_config is not None else ())
        all_column_idxs = tuple(np.arange(n_features))
        passthrough_column_idxs = tuple(
            set(all_column_idxs) - set(num_column_idxs) - set(cat_column_idxs))

        assert isinstance(cat_column_idxs, tuple)
        assert isinstance(num_column_idxs, tuple)
        assert isinstance(passthrough_column_idxs, tuple)
        return cat_column_idxs, num_column_idxs, passthrough_column_idxs

    @property
    def backbone_in_features(self) -> int:
        """Return the number of input features for the first mlp backbone layer."""
        if self.cat_embeddings:
            # cat_dim = len(self.cat_column_idxs) * self.cat_embeddings.output_dim
            cat_dim = self.cat_embeddings.output_dim
        else:
            cat_dim = len(self.cat_column_idxs)

        if self.num_embeddings:
            num_dim = (len(self.num_column_idxs) *
                       self.num_embeddings.output_dim)
        else:
            num_dim = len(self.num_column_idxs)

        return cat_dim + num_dim + len(self.passthrough_column_idxs)

    def forward(self,
                x: torch.Tensor  #
                ) -> torch.Tensor:
        if self.cat_embeddings or self.num_embeddings:

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
            x = x[:, self.cat_column_idxs + self.num_column_idxs + self.passthrough_column_idxs]

        x = self.backbone(x)
        if not self.headless:
            x = self.head(x)
        return x
