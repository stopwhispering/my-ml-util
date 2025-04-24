from typing import Sequence, Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from xgboost import DMatrix

import logging

from xgboost.callback import EvaluationMonitor

from .base_trainer import Trainer

logger = logging.getLogger(__name__)


class XGBTrainer(Trainer):
    def __init__(
        self,
        params: dict,
        early_stop=False,
        silent=False,
        use_gpu=False,
        log_transform_targets: bool = False,
        remove_columns: Sequence[str] = ("ID",),
        sample_weights: pd.Series
        | None = None,  # must have same index as df_train in fit()
        # predict_test_as_ranks=False,
        idx_outliers: pd.Index = None,
        scoring_fn: callable = None,
        fn_add_target_encoding: Callable | None = None,
        log_evaluation: int = False,  # e.g. 100 to log progress every 100 iterations
    ):
        super().__init__(
            # n_splits=n_splits,
            silent=silent,
            early_stop=early_stop,
            # preprocessor=preprocessor,
            # stratified=stratified,
            # use_gpu=use_gpu,
            log_transform_targets=log_transform_targets,
            # target_encode_columns=target_encode_columns,
            # early_stop=early_stop,
            # key_columns=key_columns,
            remove_columns=remove_columns,
            sample_weights=sample_weights,
            # predict_test_as_ranks=predict_test_as_ranks,
            idx_outliers=idx_outliers,
            scoring_fn=scoring_fn,
            fn_add_target_encoding=fn_add_target_encoding
        )
        self.params_xgb = params
        self.use_gpu = use_gpu

        self.log_evaluation = log_evaluation

        # self.early_stop = early_stop

        if self.early_stop and "early_stopping_rounds" not in params:
            params["early_stopping_rounds"] = 50

        # for xgb, there's a default max of 100 rounds for early stopping
        if self.early_stop and "n_estimators" not in params:
            params["n_estimators"] = 2000

        if not self.early_stop and "early_stopping_rounds" in params:
            logger.warning(
                "early_stopping_rounds is set but early_stop is False. Removing early_stopping_rounds"
            )
            del params["early_stopping_rounds"]

    def train_on_full(
        self,
        df_train: pd.DataFrame,
        ser_targets_train: pd.Series,
        df_test: pd.DataFrame,
    ) -> pd.DataFrame:
        if "early_stopping_rounds" in self.params_xgb:
            del self.params_xgb["early_stopping_rounds"]
        if self.early_stop:
            self.early_stop = False
        df_test_predictions = super().train_on_full(
            df_train=df_train,
            ser_targets_train=ser_targets_train,
            df_test=df_test,
        )
        return df_test_predictions

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
        if early_stop and (x_val is None or y_val is None):
            raise ValueError("Validation data is required for early stopping")

        if self.log_transform_targets:
            y_train = np.log1p(y_train)
            if y_val is not None:
                y_val = np.log1p(y_val)

        if (
            any(c for c in x_train.dtypes if c == "category")
            and "enable_categorical" not in self.params_xgb
        ):
            if not self.silent:
                print(
                    "Detected 'category' feature. Enabling enable_categorical for XGB. "
                )
            self.params_xgb["enable_categorical"] = True

        if self.use_gpu:
            # import cupy as cp  # TODOOOOOOOO

            estimator = xgb.XGBRegressor(
                **self.params_xgb,
                device="cuda",
                tree_method="hist",
            )
            # x_train = cp.array(x_train)  # noqa
        else:
            estimator = xgb.XGBRegressor(
                **self.params_xgb,
            )
        if early_stop:
            callbacks = xgb.callback.EvaluationMonitor(period=self.log_evaluation) if self.log_evaluation else []
            estimator.fit(
                X=x_train,
                y=y_train,
                eval_set=[(x_val, y_val)],
                verbose=False,
                sample_weight=ser_sample_weights_fold,  # default: None
                callbacks=callbacks,
            )
            return estimator, estimator.best_iteration
        else:
            estimator.fit(X=x_train, y=y_train, verbose=False)
            return estimator, 0

    def _predict(
        self,
        estimator: xgb.XGBRegressor,
        df_features: pd.DataFrame,
    ) -> np.ndarray:
        """same results, maybe faster (GPU)"""

        booster = estimator.get_booster()

        if any(c for c in df_features.dtypes if c == "category"):
            if self.early_stop:
                arr_pred = booster.predict(
                    DMatrix(df_features, enable_categorical=True),
                    iteration_range=(0, estimator.best_iteration + 1),
                )
            else:
                arr_pred = booster.predict(
                    DMatrix(df_features, enable_categorical=True)
                )
        else:
            if self.early_stop:
                arr_pred = booster.predict(
                    DMatrix(df_features),
                    iteration_range=(0, estimator.best_iteration + 1),
                )
            else:
                arr_pred = booster.predict(DMatrix(df_features))
        if self.log_transform_targets:
            arr_pred = np.expm1(arr_pred)

        return arr_pred
