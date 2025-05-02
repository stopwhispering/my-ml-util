import logging
from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator

from podcast.trainers.base_trainer import Trainer

logger = logging.getLogger(__name__)



class CatBoostTrainer(Trainer):
    def __init__(
        self,
        params: dict,
        cat_features: list[str] | str = "auto",
        early_stop=True,
        early_stopping_rounds=30,
        use_gpu=False,
        log_transform_targets=False,
        remove_columns: Sequence[str] = ("ID",),
        silent=False,
        sample_weights: pd.Series
        | None = None,  # must have same index as df_train in fit()
        idx_outliers: pd.Index | None = None,
        scoring_fn: callable = None,
    ):
        super().__init__(
            silent=silent,
            early_stop=early_stop,
            log_transform_targets=log_transform_targets,
            remove_columns=remove_columns,
            sample_weights=sample_weights,
            idx_outliers=idx_outliers,
            scoring_fn=scoring_fn,
        )
        self.params_catboost = params
        self.use_gpu = use_gpu
        self.early_stopping_rounds = early_stopping_rounds
        self.params_catboost["task_type"] = "GPU" if use_gpu else "CPU"
        # self.early_stop = early_stop

        self.cat_features = cat_features

        if "allow_writing_files" not in self.params_catboost:
            self.params_catboost["allow_writing_files"] = False

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._prepare(df)
        if self.cat_features == "auto":
            cat_features = list(
                df.select_dtypes(["category", "object", "string"]).columns
            )
        else:
            cat_features = self.cat_features

        if cat_features:
            df[cat_features] = df[cat_features].astype("string").fillna("NA")
        # for col in cat_features:
        #     df[col] = df[col].astype("str")  # catboost is a total mess with categories
        #     if df[col].isna().any():
        #         df[col].astype("str").fillna("NA")
        return df

    def _fit_model(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        early_stop: bool = True,
        y_train_2=None,  # for special cases like survival analysis
        y_val_2=None,  # for special cases like survival analysis
        ser_sample_weights_fold=None,  # currently only relevant for LGBM
    ) -> tuple[BaseEstimator, int]:
        assert y_train_2 is None, "not implemented"
        if early_stop and (x_val is None or y_val is None):
            raise ValueError("Validation data is required for early stopping")

        params_catboost = self.params_catboost.copy()
        if not early_stop and "use_best_model" in params_catboost:
            del params_catboost["use_best_model"]
            logger.warning(
                "use_best_model is set but early_stop is False. Removing use_best_model"
            )

        estimator = CatBoostRegressor(**params_catboost)

        if self.log_transform_targets:
            y_train = np.log1p(y_train)
            y_val = np.log1p(y_val)

        if self.cat_features == "auto":
            cat_features = list(
                x_train.select_dtypes(["category", "object", "string"]).columns
            )
        else:
            cat_features = self.cat_features

        if early_stop:
            estimator.fit(
                X=x_train,
                y=y_train,
                eval_set=[(x_val, y_val)],
                verbose=False,
                cat_features=cat_features,
                early_stopping_rounds=self.early_stopping_rounds,
                sample_weight=ser_sample_weights_fold,  # default: None
            )
            return estimator, estimator.get_best_iteration()  # noqa
        else:
            estimator.fit(
                X=x_train,
                y=y_train,
                verbose=False,
                cat_features=cat_features,
                sample_weight=ser_sample_weights_fold,  # default: None
            )
            return estimator, 0  # noqa
