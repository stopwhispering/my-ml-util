from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, x, y):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.drop(self.cols, axis=1)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str]):
        super().__init__()
        self.cols = cols
        self.cols_frequency_encoded = [f"FE_{c}" for c in self.cols]
        self.value_counts_maps = []

    def fit(self, df: pd.DataFrame):
        self.value_counts_maps.clear()
        for column in self.cols:
            self.value_counts_maps.append(
                df[column]
                .value_counts(
                    dropna=True,
                    normalize=True,
                    # relative frequency, not absolute numbers
                )
                .to_dict()
            )
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        results = []
        for i, column in enumerate(self.cols):
            ser_frequencies = x[column].map(self.value_counts_maps[i])
            results.append(ser_frequencies)
        df_results = pd.concat(results, axis=1).values
        x[self.cols_frequency_encoded] = df_results
        return x
