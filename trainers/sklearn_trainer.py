from typing import Sequence, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import logging

from podcast.trainers.base_trainer import Trainer, HasPredict

logger = logging.getLogger(__name__)



class SklearnTrainer(Trainer):
    def __init__(
        self,
        model_class,  # e.g. LogisticRegression,
        params: dict,
        early_stop=False,
        silent=False,
        log_transform_targets: bool = False,
        remove_columns: Sequence[str] = ("ID",),
        sample_weights: pd.Series
        | None = None,  # must have same index as df_train in fit()
        # predict_test_as_ranks=False,
        idx_outliers: pd.Index | None = None,
        scoring_fn: callable = None,
        fn_add_target_encoding: Callable | None = None,
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
            fn_add_target_encoding=fn_add_target_encoding,
        )
        self.params_etc = params

        if early_stop:
            raise ValueError("Early stopping is not supported for LogisticRegression")

        self.model_class = model_class
        # self.early_stop = early_stop

    def train_on_full(
        self,
        df_train: pd.DataFrame,
        ser_targets_train: pd.Series,
        df_test: pd.DataFrame,
    ) -> pd.DataFrame:
        if "early_stopping_rounds" in self.params_etc:
            del self.params_etc["early_stopping_rounds"]
        if self.early_stop:
            self.early_stop = False
        df_test_predictions = super().train_on_full(
            df_train=df_train,
            ser_targets_train=ser_targets_train,
            df_test=df_test,
        )
        return df_test_predictions

    def _predict(
        self,
        estimator: HasPredict,
        df_features: pd.DataFrame,
    ) -> np.ndarray:
        """override since sklearn api returns 0/1 for binary classification if using predict()"""
        if getattr(estimator, "predict_proba", False):
            arr_pred = estimator.predict_proba(df_features)[:, 1]  # noqa
        else:
            arr_pred = estimator.predict(df_features)
        if self.log_transform_targets:
            # arr_pred = np.exp(arr_pred)
            arr_pred = np.expm1(arr_pred)

        return arr_pred

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
            # if y_val is not None:
            #     y_val = np.log1p(y_val)

        # estimator = LogisticRegression(
        estimator = self.model_class(
            **self.params_etc,
        )
        if early_stop:
            raise ValueError("Early stopping is not supported for LogisticRegression")
        else:
            estimator.fit(X=x_train, y=y_train)
            return estimator, 0
