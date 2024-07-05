import pandas as pd
import numpy as np


def get_highly_correlated_features(df: pd.DataFrame, threshold: float = 1) -> list[str]:
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return [column for column in upper.columns if any(upper[column] >= threshold)]
