from my_ml_util.mlp import MLP, seed, BackboneConfig, HeadConfig, NumericalEmbeddingsConfig
import pandas as pd
import torch

from my_ml_util.mlp.categorical_embeddings import CategoricalEmbeddingsConfig

device = 'cpu'


def test_simple_mlp():
    df_train = pd.DataFrame({'a': [1, 2, 3],
                             'b': [4, 5, 6]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
        n_features=df_train.shape[1],
        backbone_config=BackboneConfig(
            out_features_per_layer=[10, 5],
            dropout=0.1,
        ),
        head_config=HeadConfig(dim_out=ser_targets_train.nunique()),  # multi-class classification
    ).to(device)
    model.eval()
    X = torch.tensor(df_train.values, dtype=torch.float32).to(device)
    pred = model(X)
    print(pred)


def test_mlp_with_categorical_embeddings():
    df_train = pd.DataFrame({'a': [0, 1, 2, 3],  # categorical must be ordinal encoded (i.e. 0..n-1)
                             'b': [0, 2, 1, 0],
                             'c': [7.1, 8.2, 9.4, 13.1],
                             'd': [0, 0, 0, 0],
                             'e': [0, 0, 1, 2]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
        n_features=df_train.shape[1],
        backbone_config=BackboneConfig(
            out_features_per_layer=[10, 5],
            dropout=0.1,
        ),
        categorical_embeddings_config=CategoricalEmbeddingsConfig(
            column_idxs=(0, 1, 3, 4),
            cat_column_n_unique_values=(4, 3, 1, 3),
        ),
        head_config=HeadConfig(dim_out=ser_targets_train.nunique()),  # multi-class classification
    ).to(device)
    model.eval()
    X = torch.tensor(df_train.values, dtype=torch.float32).to(device)
    pred = model(X)
    print(pred)


def test_mlp_l():
    df_train = pd.DataFrame({'a': [0, 1, 2],  # categorical must be ordinal encoded (i.e. 0..n-1)
                             'b': [0, 2, 1],
                             'c': [7.1, 8.2, 9.4],
                             'd': [3.1, 0.2, 3.9],
                             'e': [0, 0, 1]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
        n_features=df_train.shape[1],
        backbone_config=BackboneConfig(
            out_features_per_layer=[10, 5],
            dropout=0.1,
        ),
        head_config=HeadConfig(dim_out=ser_targets_train.nunique()),  # multi-class classification
        numerical_embeddings_config=NumericalEmbeddingsConfig(
            architecture='MLP-L',
            column_idxs=(2, 3),
            linear_embedding_dim=32,
        ),
    ).to(device)
    model.eval()
    X = torch.tensor(df_train.values, dtype=torch.float32).to(device)
    pred = model(X)
    print(pred)


def test_mlp_lr():
    df_train = pd.DataFrame({'a': [0, 1, 2],  # categorical must be ordinal encoded (i.e. 0..n-1)
                             'b': [0, 2, 1],
                             'c': [7.1, 8.2, 9.4],
                             'd': [3.1, 0.2, 3.9],
                             'e': [0, 0, 1]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
        n_features=df_train.shape[1],
        backbone_config=BackboneConfig(
            out_features_per_layer=[10, 5, 3, 2, 5],
            dropout=0.1,
        ),
        head_config=HeadConfig(dim_out=ser_targets_train.nunique()),  # multi-class classification
        numerical_embeddings_config=NumericalEmbeddingsConfig(
            architecture='MLP-LR',
            column_idxs=(2, 3),
            linear_embedding_dim=28,
        ),
        categorical_embeddings_config=CategoricalEmbeddingsConfig(
            cat_column_n_unique_values=(3, 3, 2),
            column_idxs=(0, 1, 4),
        ),
    ).to(device)
    model.eval()
    X = torch.tensor(df_train.values, dtype=torch.float32).to(device)
    pred = model(X)
    print(pred)


def test_mlp_p():
    df_train = pd.DataFrame({'a': [0, 1, 2],  # categorical must be ordinal encoded (i.e. 0..n-1)
                             'b': [0, 2, 1],
                             'c': [7.1, 8.2, 9.4],
                             'd': [3.1, 0.2, 3.9],
                             'e': [0, 0, 1]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
        n_features=df_train.shape[1],
        backbone_config=BackboneConfig(
            out_features_per_layer=[10, 5],
            dropout=0.1,
        ),
        head_config=HeadConfig(dim_out=ser_targets_train.nunique()),  # multi-class classification
        numerical_embeddings_config=NumericalEmbeddingsConfig(
            architecture='MLP-P',
            column_idxs=(2, 3),
            periodic_embedding_dim=30,
            periodic_sigma=0.11,
        ),
    ).to(device)
    model.eval()
    X = torch.tensor(df_train.values, dtype=torch.float32).to(device)
    pred = model(X)
    print(pred)


def test_mlp_pl():
    df_train = pd.DataFrame({'a': [0, 1, 2],  # categorical must be ordinal encoded (i.e. 0..n-1)
                             'b': [0, 2, 1],
                             'c': [7.1, 8.2, 9.4],
                             'd': [3.1, 0.2, 3.9],
                             'e': [0, 0, 1]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
        n_features=df_train.shape[1],
        backbone_config=BackboneConfig(
            out_features_per_layer=[10], #, 5],
            dropout=0.1,
        ),
        head_config=HeadConfig(dim_out=ser_targets_train.nunique()),  # multi-class classification
        numerical_embeddings_config=NumericalEmbeddingsConfig(
            architecture='MLP-PL',
            column_idxs=(2, 3),
            periodic_embedding_dim=30,
            periodic_sigma=0.11,
            linear_embedding_dim=35,
        ),
        categorical_embeddings_config=CategoricalEmbeddingsConfig(
            column_idxs=(0, 1, 4),
            cat_column_n_unique_values=(3, 3, 2),
        ),
    ).to(device)
    model.eval()
    X = torch.tensor(df_train.values, dtype=torch.float32).to(device)
    pred = model(X)
    print(pred)


def test_mlp_plr():
    df_train = pd.DataFrame({'a': [0, 1, 2],  # categorical must be ordinal encoded (i.e. 0..n-1)
                             'b': [0, 2, 1],
                             'c': [7.1, 8.2, 9.4],
                             'd': [3.1, 0.2, 3.9],
                             'e': [0, 0, 1]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
        n_features=df_train.shape[1],
        backbone_config=BackboneConfig(
            out_features_per_layer=[10, 5],
            dropout=0.1,
        ),
        head_config=HeadConfig(dim_out=ser_targets_train.nunique()),  # multi-class classification
        numerical_embeddings_config=NumericalEmbeddingsConfig(
            architecture='MLP-PLR',
            column_idxs=(2, 3),
            periodic_embedding_dim=30,
            periodic_sigma=0.11,
            linear_embedding_dim=35,
        ),
        categorical_embeddings_config=CategoricalEmbeddingsConfig(
            cat_column_n_unique_values=(3, 3),
            column_idxs=(0, 1),
        ),
        # out_features_per_layer=[10, 5],
        # dropout=0.1,
    ).to(device)
    model.eval()
    X = torch.tensor(df_train.values, dtype=torch.float32).to(device)
    pred = model(X)
    print(pred)


def test_mlp_batch_normalization():
    df_train = pd.DataFrame({'a': [0, 1, 2],  # categorical must be ordinal encoded (i.e. 0..n-1)
                             'b': [0, 2, 1],
                             'c': [7.1, 8.2, 9.4],
                             'd': [3.1, 0.2, 3.9],
                             'e': [0, 0, 1]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
        n_features=df_train.shape[1],
        backbone_config=BackboneConfig(
            out_features_per_layer=[10, 5],
            dropout=0.1,
            apply_batch_normalization=True,  # default: False
            apply_layer_normalization=True,  # default: False  (makes no sense together, but technically ok)
        ),
        head_config=HeadConfig(dim_out=ser_targets_train.nunique()),  # multi-class classification
        numerical_embeddings_config=NumericalEmbeddingsConfig(
            architecture='MLP-PLR',
            column_idxs=(2, 3),
            periodic_embedding_dim=30,
            periodic_sigma=0.11,
            linear_embedding_dim=35,
        ),
        categorical_embeddings_config=CategoricalEmbeddingsConfig(
            cat_column_n_unique_values=(3, 3),
            column_idxs=(0, 1),
        ),
        # out_features_per_layer=[10, 5],
        # dropout=0.1,
    ).to(device)
    model.eval()
    X = torch.tensor(df_train.values, dtype=torch.float32).to(device)
    pred = model(X)
    print(pred)


def test_ad_hoc():
    df_train = pd.DataFrame({'a': [0, 1, 2],  # categorical must be ordinal encoded (i.e. 0..n-1)
                             'b': [0, 2, 1],
                             'c': [7.1, 8.2, 9.4],
                             'd': [3.1, 0.2, 3.9],
                             'e': [0, 0, 1]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
        n_features=df_train.shape[1],
        backbone_config=BackboneConfig(
            out_features_per_layer=[412, 689, 345, 465, 613, 554],
            dropout=0.116305,
        ),
        head_config=HeadConfig(dim_out=ser_targets_train.nunique()),  # multi-class classification
        numerical_embeddings_config=NumericalEmbeddingsConfig(
            architecture='MLP-LR',
            column_idxs=(2, 3),
            periodic_embedding_dim=30,
            periodic_sigma=0.11,
            linear_embedding_dim=35,
        ),
        # categorical_embeddings_config=CategoricalEmbeddingsConfig(
        #     cat_column_n_unique_values=(3, 3),
        #     column_idxs=(0, 1),
        # ),
        # out_features_per_layer=[10, 5],
        # dropout=0.1,
    ).to(device)
    model.eval()
    X = torch.tensor(df_train.values, dtype=torch.float32).to(device)
    pred = model(X)
    print(pred)


seed(42)
# test_simple_mlp()
# test_mlp_with_categorical_embeddings()
# test_mlp_l()
# test_mlp_lr()
# test_mlp_p()
# test_mlp_pl()
# test_mlp_plr()
test_mlp_batch_normalization()

# test_ad_hoc()

