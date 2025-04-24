from typing import Sequence, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import logging

from .base_trainer import Trainer

logger = logging.getLogger(__name__)


class LGBMTrainer(Trainer):
    def __init__(
        self,
        params: dict,
        scoring_fn: Callable,
        silent=False,
        early_stop=False,
        stopping_rounds=20,
        use_gpu=False,
        log_transform_targets: bool = False,
        remove_columns: Sequence[str] = ("ID",),
        sample_weights: pd.Series
        | None = None,  # must have same index as df_train in fit()
        idx_outliers: pd.Index | None = None,
        task_type="regression",  # "regression" or "classification"
        fn_add_target_encoding: Callable | None = None,
        log_evaluation: int = False,  # e.g. 100 to log progress every 100 iterations
    ):
        assert use_gpu is False, "TODO IMPLEMENT GPU FOR LBGM"

        super().__init__(
            silent=silent,
            early_stop=early_stop,
            log_transform_targets=log_transform_targets,
            remove_columns=remove_columns,
            sample_weights=sample_weights,
            idx_outliers=idx_outliers,
            scoring_fn=scoring_fn,
            task_type=task_type,
            fn_add_target_encoding=fn_add_target_encoding,
        )

        self.params_lgbm = params
        self.task_type = task_type
        self.log_evaluation = log_evaluation
        self.stopping_rounds = stopping_rounds

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
        if self.task_type == 'regression':
            lgbm = lgb.LGBMRegressor(
                **self.params_lgbm,
            )
        elif self.task_type == 'classification':
            lgbm = lgb.LGBMClassifier(
                **self.params_lgbm,
            )
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        if self.log_transform_targets:
            y_train = np.log1p(y_train)
            # y_train = np.log(y_train)
            if y_val is not None:
                y_val = np.log1p(y_val)
                # y_val = np.log(y_val)

        if early_stop:
            early_stopping_callback = lgb.early_stopping(self.stopping_rounds, verbose=False)
            callbacks = [early_stopping_callback] if not self.log_evaluation else [early_stopping_callback, lgb.log_evaluation(self.log_evaluation)]
            lgbm.fit(
                X=x_train,
                y=y_train,
                eval_set=[(x_val, y_val)],
                callbacks=callbacks,
                sample_weight=ser_sample_weights_fold,  # default: None
                # eval_metric="mape",  # todo
                # eval_metric=self.custom_eval_metric,
                # if not None, params must incl. 'metric': 'custom'
            )
            return lgbm, lgbm.best_iteration_
        else:
            lgbm.fit(
                X=x_train,
                y=y_train,
                sample_weight=ser_sample_weights_fold,  # default: None
            )
            return lgbm, 0
