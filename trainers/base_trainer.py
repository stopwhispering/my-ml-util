from abc import ABC, abstractmethod
from typing import Protocol, Iterable, Callable
import shap

import pandas as pd
import numpy as np

import sklearn
import logging

from sklearn.model_selection import KFold

sklearn.set_config(transform_output="pandas")

logger = logging.getLogger(__name__)


class HasPredict(Protocol):
    def predict(self, X) -> np.ndarray: ...


class Trainer(ABC):
    def __init__(
            self,
            # n_splits: int,
            # preprocessor: Preprocessor,
            silent,
            # use_gpu,
            early_stop,
            # key_columns,
            remove_columns,
            # splitter,  # KFold, GroupKFold, etc.
            scoring_fn: Callable,
            log_transform_targets: bool = False,
            sample_weights: pd.Series
                            | None = None,  # must have same index as df_train in fit()
            idx_outliers: pd.Index | None = None,
            task_type='regression',  # "regression" or "classification"
            fn_add_target_encoding: Callable | None = None,

    ):
        # self.log_transform_targets = log_transform_targets
        # self.n_splits = n_splits
        self.silent = silent
        # self.stratified = stratified
        # self.target_encode_columns = target_encode_columns
        # self.preprocessor = preprocessor

        self.log_transform_targets = log_transform_targets
        self.early_stop = early_stop
        self.fn_add_target_encoding = fn_add_target_encoding

        # self.use_gpu = use_gpu
        # self.debug = debug
        # self.splitter = splitter
        # if self.splitter is None:
        #     self.splitter = sklearn.model_selection.KFold(
        #         n_splits=5, shuffle=True, random_state=42
        #     )

        self.remove_columns = remove_columns
        self.fitted_cv_models = []

        self.sample_weights = sample_weights
        self.idx_outliers = idx_outliers

        self.scoring_fn = scoring_fn
        self.task_type = task_type

    @abstractmethod
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
    ) -> tuple[HasPredict, int]:
        """implement for LGBM, XGBoost, CatBoost, etc.
        returns the trained model and the best iteration number"""
        pass

    def remove_outliers(
            self, df: pd.DataFrame, ser_targets: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        idx_outliers_in_fold = self.idx_outliers.intersection(df.index)
        df = df.drop(idx_outliers_in_fold, axis=0)
        ser_targets = ser_targets.drop(idx_outliers_in_fold, axis=0)
        return df, ser_targets

    def train_on_full(
            self,
            df_train: pd.DataFrame,
            ser_targets_train: pd.Series,
            df_test: pd.DataFrame,
    ) -> pd.DataFrame:
        logger.debug(f"{df_train.shape=}")
        logger.debug(f"{ser_targets_train.shape=}")
        logger.debug(f"{df_test.shape=}")

        df_train = df_train.copy()
        df_train = self._prepare(df_train)

        logger.debug("Starting model training.")
        if self.idx_outliers is not None:
            df_train, ser_targets_train = self.remove_outliers(
                df_train, ser_targets_train
            )
        estimator, _best_iteration = self._fit_model(
            x_train=df_train,
            y_train=ser_targets_train,
            early_stop=False,
        )

        arr_pred_train = self._predict(estimator, df_train)
        # score_train = roc_auc_score(ser_targets_train, arr_pred_train)
        score_train = self.scoring_fn(ser_targets_train, arr_pred_train)
        logger.info(f"Score on Train (TRAIN! LEAK!): {score_train=:.5f}")

        logger.debug(f"Starting test prediction.")
        df_test_predictions = self.predict_with_fitted_models(df_test, [estimator])

        self.fitted_cv_models.append(estimator)

        return df_test_predictions

    def predict_with_fitted_models(
            self,
            df_test: pd.DataFrame,
            fitted_models: list | None = None,
            list_df_test_with_te: list[pd.DataFrame] | None = None,  # only relevant if fn_add_target_encoding is not None
    ) -> pd.DataFrame:
        fitted_models = fitted_models or self.fitted_cv_models

        df_test = df_test.copy()
        df_test = self._prepare(df_test)
        df_test_original = df_test.copy()

        arr_test_predictions = np.zeros(len(df_test))

        for col in self.remove_columns:
            if col in df_test.columns:
                df_test = df_test.drop(col, axis=1)

        for i, estimator in enumerate(fitted_models):
            if list_df_test_with_te is not None and len(list_df_test_with_te) > 0:
                df_test = list_df_test_with_te[i]
            arr_test_predictions += self._predict(estimator, df_test)

        arr_test_predictions /= len(fitted_models)

        # we return the test preds with original index for visualization etc.
        df_test_predictions = df_test_original.copy().assign(pred=arr_test_predictions)

        return df_test_predictions

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """override if required"""
        for col in self.remove_columns:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df

    def train_and_score(
            self,
            df_train: pd.DataFrame,
            ser_targets_train: pd.Series,
            df_test: pd.DataFrame | None = None,
            print_score=False,
            ser_targets_train_2: pd.Series
                                 | None = None,  # for special cases like survival analysis
            compute_shap_values=False,
            splits: Iterable = None,  # default: KFold 5
    ) -> tuple[
        float,
        list[int],
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        shap.Explainer | None,
    ]:
        best_iterations: list[int] = []
        arr_oof_predictions_all_folds = np.zeros(len(ser_targets_train))

        # for visualization, we keep the val. predictions for each fold
        df_pred_val_folds = ser_targets_train.to_frame()
        assert (df_pred_val_folds.index == df_train.index).all()  # noqa
        # df_pred_val_folds[self.key_columns] = df_train[self.key_columns]

        df_train = df_train.copy()
        df_train = self._prepare(df_train)

        # splits = self.splitter.split(df_train, groups=df_train['date'].dt.year)
        if splits is None:
            splits = KFold(n_splits=5, shuffle=True, random_state=42).split(df_train)

        # shap_values_per_fold = []
        # val_indices = []
        df_shap_values = pd.DataFrame(index=df_train.index, columns=df_train.columns)
        list_df_test_with_te = []  # only relevant if fn_add_target_encoding is not None
        for fold_no, (train_idx, val_idx) in enumerate(splits):
            # info_train = f"{df_train.loc[train_idx]['date'].min().strftime('%Y-%m-%d')} - {df_train.loc[train_idx]['date'].max().strftime('%Y-%m-%d')}"
            # info_train = f"{df_train.iloc[train_idx]['date'].min().strftime('%Y-%m-%d')} - {df_train.iloc[train_idx]['date'].max().strftime('%Y-%m-%d')}"
            # info_valid = f"{df_train.loc[val_idx]['date'].min().strftime('%Y-%m-%d')} - {df_train.loc[val_idx]['date'].max().strftime('%Y-%m-%d')}"
            # info_valid = f"{df_train.iloc[val_idx]['date'].min().strftime('%Y-%m-%d')} - {df_train.iloc[val_idx]['date'].max().strftime('%Y-%m-%d')}"
            # logger.info(
            #     f"Fold {fold_no}: Train: {info_train} ({len(train_idx) :_}) / Valid: {info_valid} ({len(val_idx) :_}))"
            # )

            df_train_fold = df_train.iloc[train_idx]  # .reset_index(drop=True)
            # df_train_fold = df_train.loc[train_idx]  # .reset_index(drop=True)
            df_val_fold = df_train.iloc[val_idx]  # .reset_index(drop=True)
            # df_val_fold = df_train.loc[val_idx]  # .reset_index(drop=True)

            # ser_targets_train_fold: pd.Series = ser_targets_train.loc[
            #     train_idx
            # ]  # .reset_index(drop=True)
            ser_targets_train_fold: pd.Series = ser_targets_train.iloc[
                train_idx
            ]  # .reset_index(drop=True)
            # ser_targets_val_fold: pd.Series = ser_targets_train.loc[
            #     val_idx
            # ]  # .reset_index(drop=True)
            ser_targets_val_fold: pd.Series = ser_targets_train.iloc[
                val_idx
            ]  # .reset_index(drop=True)

            if ser_targets_train_2 is not None:
                ser_targets_train_2_fold = ser_targets_train_2.iloc[train_idx]
                ser_targets_val_2_fold = ser_targets_train_2.iloc[val_idx]
            else:
                ser_targets_train_2_fold = None
                ser_targets_val_2_fold = None

            if self.sample_weights is not None:
                ser_sample_weights_fold = self.sample_weights.iloc[train_idx]
            else:
                ser_sample_weights_fold = None

            if self.fn_add_target_encoding:
                # if we are applying target encoding, we need to compute te for each
                # fold separately; this applies to test set as well
                df_train_fold, df_val_fold, df_test_with_fold_te = self.fn_add_target_encoding(
                    df_train=df_train_fold.copy(),
                    df_val=df_val_fold.copy(),
                    df_test=df_test.copy() if df_test is not None else None,
                    ser_targets_train=ser_targets_train_fold,
                    i_fold=fold_no,
                )
                list_df_test_with_te.append(df_test_with_fold_te)

            logger.debug(f"{df_train_fold.shape=}")
            logger.debug(f"{df_val_fold.shape=}")
            logger.debug("Starting preprocessing.")

            logger.debug(f"{df_train_fold.shape=}")
            logger.debug(f"{df_val_fold.shape=}")

            logger.debug("Starting model training.")
            # temp = df_val_fold[[c for c in df_val_fold.columns if c.startswith('year_')]]

            if self.idx_outliers is not None:
                df_train_fold, ser_targets_train_fold = self.remove_outliers(
                    df_train_fold, ser_targets_train_fold
                )
            estimator, best_iteration = self._fit_model(
                x_train=df_train_fold,
                y_train=ser_targets_train_fold,
                x_val=df_val_fold,
                y_val=ser_targets_val_fold,
                early_stop=self.early_stop,
                y_train_2=ser_targets_train_2_fold,
                y_val_2=ser_targets_val_2_fold,
                ser_sample_weights_fold=ser_sample_weights_fold,
            )
            logger.debug("Finished model training.")
            best_iterations.append(best_iteration)
            self.fitted_cv_models.append(estimator)

            arr_pred_val_fold = self._predict(estimator, df_val_fold)
            logger.debug(f"{arr_pred_val_fold=}")
            arr_oof_predictions_all_folds[val_idx] = arr_pred_val_fold
            df_pred_val_folds[f"fold_{fold_no}"] = np.nan
            # df_pred_val_folds.loc[
            #     df_pred_val_folds.index[val_idx], f"fold_{fold_no}"
            # ] = arr_pred_val_fold
            # df_pred_val_folds.loc[val_idx, f"fold_{fold_no}"] = arr_pred_val_fold
            # df_pred_val_folds.iat[val_idx], f"fold_{fold_no}"] = arr_pred_val_fold
            # pandas can be so weird...
            df_pred_val_folds.iloc[
                val_idx, df_pred_val_folds.columns.get_loc(f"fold_{fold_no}")
            ] = arr_pred_val_fold

            arr_pred_val_fold = pd.Series(arr_pred_val_fold).fillna(0).values
            if ser_targets_val_fold.isna().any():
                raise ValueError("NaNs in validation targets")

            if np.isnan(arr_pred_val_fold).any():
                raise ValueError("NaNs STILLLLLL TODO in validation predictions")

            # score = roc_auc_score(ser_targets_val_fold, arr_pred_val_fold)
            score = self.scoring_fn(ser_targets_val_fold, arr_pred_val_fold)
            logger.info(
                f"Fold {fold_no}: {score=:.5f} (best iteration: {best_iteration})"
            )
            # print(fold_no, mape_fold)

            if compute_shap_values:
                # this would be better, but background data does not allow for categorical features
                # shap_values_fold = shap.TreeExplainer(estimator, df_train_fold).shap_values(
                #     df_val_fold
                # )  # same shape as df_val_fold

                try:
                    shap_values_fold = shap.TreeExplainer(estimator).shap_values(
                        df_val_fold
                    )
                except shap.utils._exceptions.ExplainerError:  # noqa
                    shap_values_fold = shap.TreeExplainer(estimator).shap_values(
                        df_val_fold, check_additivity=False
                    )
                    if not self.silent:
                        print("Warning: check_additivity=False was required for SHAP")

                df_shap_values.iloc[val_idx] = shap_values_fold
                # shap_values_per_fold.append(
                #     shap.TreeExplainer(estimator).shap_values(df_val_fold)
                # )
                # val_indices.append(val_idx)

        arr_oof_predictions = arr_oof_predictions_all_folds
        # score = roc_auc_score(ser_targets_train, arr_oof_predictions)
        score = self.scoring_fn(ser_targets_train, arr_oof_predictions)

        # return oof preds with original index and key columns for visualization etc.
        df_oof_predictions = df_train.copy().assign(pred=arr_oof_predictions)

        logger.info(f"{'Score':<28}: {score=:.5f}")
        if print_score:
            print(f"{'Score':<28}: {score=:.5f} ")

        # df_test = self._prepare(df_test)
        df_test_predictions = (
            self.predict_with_fitted_models(df_test,
                                            list_df_test_with_te=list_df_test_with_te
                                            ) if df_test is not None else None
        )

        return (
            score,
            best_iterations,
            df_oof_predictions,
            df_test_predictions,
            df_pred_val_folds,
            df_shap_values.astype("float64") if compute_shap_values else None,
        )

    def _predict(
            self,
            estimator: HasPredict,
            df_features: pd.DataFrame,
    ) -> np.ndarray:
        """override for Classifiers that require postprocessing"""
        if self.task_type == 'regression':
            arr_pred = estimator.predict(df_features)
        elif self.task_type == 'classification':
            arr_pred = estimator.predict_proba(df_features)[:, 1]
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        if self.log_transform_targets:
            # arr_pred = np.exp(arr_pred)
            arr_pred = np.expm1(arr_pred)

        return arr_pred

# class RFTrainer(Trainer):
#     def __init__(
#         self,
#         log_transform_targets: bool,
#         n_splits: int,
#         params_rf: dict,
#     ):
#         super().__init__(log_transform_targets=log_transform_targets, n_splits=n_splits)
#         self.params_rf = params_rf
#
#     def fit_model(
#         self,
#         x_train: pd.DataFrame,
#         y_train: pd.Series,
#         x_val: pd.DataFrame,
#         y_val: pd.Series,
#     ) -> tuple[BaseEstimator, int]:
#         estimator = ensemble.RandomForestRegressor(
#             **self.params_rf,
#         )
#         estimator.fit(
#             X=x_train,
#             y=y_train,
#         )
#         return estimator, 0  # noqa
