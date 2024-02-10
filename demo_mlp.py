from my_ml_util.mlp import MLP, seed
import pandas as pd
import torch

device = 'cpu'


def test_simple_mlp():
    df_train = pd.DataFrame({'a': [1, 2, 3],
                             'b': [4, 5, 6]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
                # in_features=df_train.shape[1],
                out_features_per_layer=[10, 5],
                cat_column_idxs=(0, 1),
                num_column_idxs=(),
                dropout=0.1,
                dim_out=ser_targets_train.nunique()  # multi-class classification
            ).to(device)
    model.eval()
    X = torch.tensor(df_train.values, dtype=torch.float32).to(device)
    pred = model(X)
    print(pred)


def test_mlp_with_categorical_embeddings():
    df_train = pd.DataFrame({'a': [0, 1, 2],  # categorical must be ordinal encoded (i.e. 0..n-1)
                             'b': [0, 2, 1],
                             'c': [7.1, 8.2, 9.4],
                             'd': [0, 0, 0],
                             'e': [0, 0, 1]})
    ser_targets_train = pd.Series([1, 2, 3])
    model = MLP(
                # in_features=df_train.shape[1],
                out_features_per_layer=[10, 5],
                cat_column_idxs=(0, 1, 3, 4),
                num_column_idxs=(2,),
                dropout=0.1,
                dim_out=ser_targets_train.nunique(),  # multi-class classification
                use_cat_embedding=True,
                cat_embedding_use_bias=True,
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
                # in_features=df_train.shape[1],
                out_features_per_layer=[10, 5],
                dropout=0.1,
                dim_out=ser_targets_train.nunique(),  # multi-class classification
                cat_column_idxs=(0, 1, 4),
                num_column_idxs=(2, 3),
                numerical_embed_architecture='MLP-L',
                linear_embedding_dim=35,
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
                # in_features=df_train.shape[1],
                out_features_per_layer=[10, 5],
                dropout=0.1,
                dim_out=ser_targets_train.nunique(),  # multi-class classification
                cat_column_idxs=(0, 1, 4),
                num_column_idxs=(2, 3),
                numerical_embed_architecture='MLP-LR',
                linear_embedding_dim=35,
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
                # in_features=df_train.shape[1],
                out_features_per_layer=[10, 5],
                dropout=0.1,
                dim_out=ser_targets_train.nunique(),  # multi-class classification
                cat_column_idxs=(0, 1, 4),
                num_column_idxs=(2, 3),
                numerical_embed_architecture='MLP-P',
                periodic_embedding_dim=30,
                periodic_sigma=0.11,
                # linear_embedding_dim=35,
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
                # in_features=df_train.shape[1],
                out_features_per_layer=[10, 5],
                dropout=0.1,
                dim_out=ser_targets_train.nunique(),  # multi-class classification
                cat_column_idxs=(0, 1, 4),
                num_column_idxs=(2, 3),
                numerical_embed_architecture='MLP-PL',
                periodic_embedding_dim=30,
                periodic_sigma=0.11,
                linear_embedding_dim=35,
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
                # in_features=df_train.shape[1],
                out_features_per_layer=[10, 5],
                dropout=0.1,
                dim_out=ser_targets_train.nunique(),  # multi-class classification
                cat_column_idxs=(0, 1),  # , 1, 4
                passthrough_column_idxs=(),
                use_cat_embedding=True,
                num_column_idxs=(2, 3),
                numerical_embed_architecture='MLP-PLR',
                periodic_embedding_dim=30,
                periodic_sigma=0.11,
                linear_embedding_dim=35,
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
test_mlp_plr()  # [0.2136
