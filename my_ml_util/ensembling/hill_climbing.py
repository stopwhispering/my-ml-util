from typing import Callable

from sklearn.metrics import root_mean_squared_error

from podcast.constants import PATH_TRAIN, PATH_INTERIM_RESULTS
import pandas as pd
import numpy as np
import gc


def score_multiple_rmse(arr_targets: np.array, arr_predicted: np.array) -> np.array:
    """
    arr_targets has shape (n_samples,)
    arr_predicted has shape (n_samples, n_models)

    This function computes the RMSE for each model in arr_predicted.
    It returns an array of shape (n_models,) with the RMSE for each model.
    """
    scores = []
    for i in range(arr_predicted.shape[1]):
        score = root_mean_squared_error(arr_targets, arr_predicted[:, i])
        scores.append(score)
    return np.array(scores)


class HillClimber:
    """
    adapted from cdeotte implementation from https://www.kaggle.com/code/cdeotte/fast-gpu-hill-climbing-starter-cv-0-94-lb-0-94
    see there for GPU optimization
    see https://www.kaggle.com/code/stopwhispering/rainfall-56 for own GPU implementation

    """

    def __init__(
        self,
        score_multiple_function: Callable,
        higher_is_better=False,
        tol=1e-6,
        max_models=1000,
        use_negative_weights=False,
    ):
        self.score_multiple_function = score_multiple_function
        self.higher_is_better = higher_is_better
        self.tol = tol
        self.max_models = max_models
        self.use_negative_weights = use_negative_weights

    def run(self, df_pred_oof: pd.DataFrame, ser_targets):
        """
        df_pred_oof: pd.DataFrame with shape (n_samples, n_models))
        ser_targets: pd.Series with shape (n_samples,)
        """

        # compute metric for each oof
        best_score = 0 if self.higher_is_better else 1000
        best_index = -1

        for col_idx, col in enumerate(df_pred_oof.columns):
            score = root_mean_squared_error(ser_targets, df_pred_oof[col])
            if (score > best_score and self.higher_is_better) or (
                score < best_score and not self.higher_is_better
            ):
                best_score = score
                best_index = col_idx
            print(f"Score {score:0.5f} {col}")
        print()
        print(
            f"Best single model is {df_pred_oof.columns[best_index]} with Score = {best_score:0.5f}"
        )

        indices = [best_index]
        old_best_score = best_score

        x_train2 = df_pred_oof.values
        arr_best_ensemble = x_train2[:, best_index]
        arr_truth = ser_targets.values
        start = -0.50 if self.use_negative_weights else 0.01
        possible_weights = np.arange(start, 0.51, 0.01)
        n_possible_weights = len(possible_weights)

        # BEGIN HILL CLIMBING
        models = [best_index]
        weights = []
        metrics = [best_score]

        for i_iter in range(10_000_000):
            best_score = 0 if self.higher_is_better else 1000
            best_index = -1
            best_weight = 0

            # TRY ADDING ONE MORE MODEL
            for i_col, _ in enumerate(df_pred_oof.columns):
                arr_new_model = x_train2[:, i_col]
                # best current ensemble and new model are multiplied by weights (corresponding; to add up to 1)
                arr_best_ensemble_by_weights = np.repeat(
                    arr_best_ensemble[:, np.newaxis], n_possible_weights, axis=1
                ) * (1 - possible_weights)
                arr_new_model_by_weights = (
                    np.repeat(arr_new_model[:, np.newaxis], n_possible_weights, axis=1)
                    * possible_weights
                )
                # ensemble current best, using all possible weights
                arr_combined = arr_best_ensemble_by_weights + arr_new_model_by_weights
                # score each of the possible combinations (i.e. score each weight that the new model might be given)
                new_scores = self.score_multiple_function(arr_truth, arr_combined)
                new_score = (
                    np.max(new_scores).item()
                    if self.higher_is_better
                    else np.min(new_scores).item()
                )
                if (new_score > best_score and self.higher_is_better) or (
                    new_score < best_score and not self.higher_is_better
                ):
                    best_score = new_score
                    best_index = i_col
                    ii = (
                        np.argmax(new_scores).item()
                        if self.higher_is_better
                        else np.argmin(new_scores).item()
                    )
                    best_weight = possible_weights[ii].item()
                    potential_ensemble = arr_combined[:, ii]
            # del new_model, m1, m2, mm, new_aucs, new_score  #todo
            gc.collect()

            # STOPPING CRITERIA
            indices.append(best_index)
            indices = list(np.unique(indices))
            if len(indices) > self.max_models:
                print(f"=> We reached {self.max_models} models")
                # indices = indices[:-1]
                break
            if (best_score < (old_best_score + self.tol) and self.higher_is_better) or (
                best_score > (old_best_score - self.tol) and not self.higher_is_better
            ):
                print(f"=> We reached tolerance {self.tol}")
                break

            # RECORD NEW RESULT
            print(
                i_iter,
                "New best Score",
                best_score,
                f'adding "{df_pred_oof.columns[best_index]}"',
                "with weight",
                f"{best_weight:0.3f}.",
            )
            models.append(best_index)
            weights.append(best_weight)
            metrics.append(best_score)
            arr_best_ensemble = potential_ensemble  # noqa
            old_best_score = best_score

        # print(f"{i_iter=}")  # noqa

        # Above we may have added the same model more than once. So below we consolidate all updates and compute a unique weight for each OOF model.
        wgt = np.array([1])
        for w in weights:
            wgt = wgt * (1 - w)
            wgt = np.concatenate([wgt, np.array([w])])

        rows = []
        t = 0
        for m, w, s in zip(models, wgt, metrics):
            name = df_pred_oof.columns[m]
            dd = {"weight": w, "model": name}
            rows.append(dd)
            t += float(f"{w:.3f}")

        # DISPLAY WEIGHT PER MODEL
        df_weights = pd.DataFrame(rows)
        df_weights = (
            df_weights.groupby("model")
            .agg("sum")
            .reset_index()
            .sort_values("weight", ascending=False)
        )
        df_weights = df_weights.reset_index(drop=True)

        final_weights = []
        for model in df_pred_oof.columns:
            df_model = df_weights.loc[df_weights["model"] == model]
            if len(df_model) == 0:
                weight = 0
            elif len(df_model) > 1:
                raise ValueError(
                    f"Model {model} has multiple weights: {df_model}. Check implementation."
                )
            else:
                weight = df_model["weight"].values[0]
            final_weights.append(weight)

        # # if running on kaggle, display the results
        # if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", False):
        #     display(df_weights)  # noqa

        return final_weights, df_weights


if __name__ == "__main__":
    df_train = (
        pd.read_csv(PATH_TRAIN).set_index("id").drop("Listening_Time_minutes", axis=1)
    ).iloc[:100_000]
    ser_targets_train = (
        pd.read_csv(PATH_TRAIN).set_index("id")["Listening_Time_minutes"].iloc[:100_000]
    )

    np.random.seed(2)
    ser_pred_lgbm = (
        pd.read_pickle(PATH_INTERIM_RESULTS / "ser_oof_preds_lgbm.pkl")
        + np.random.normal(0, 1, size=750_000)
    ).iloc[:100_000]
    ser_pred_xgb = (
        pd.read_pickle(PATH_INTERIM_RESULTS / "ser_oof_preds_xgb.pkl")
        + np.random.normal(0, 1.2, size=750_000)
    ).iloc[:100_000]
    ser_pred_dummy = (
        pd.read_pickle(PATH_INTERIM_RESULTS / "ser_oof_preds_xgb.pkl")
        + np.random.normal(0, 1.2, size=750_000)
    ).iloc[:100_000]
    # add some noise
    ser_pred_dummy += np.random.normal(0, 0.1, size=ser_pred_dummy.shape)
    df_pred_oof = pd.concat([ser_pred_lgbm, ser_pred_xgb, ser_pred_dummy], axis=1)
    df_pred_oof.columns = ["lgbm", "xgb", "dummy"]

    final_weights, df_weights = HillClimber(
        score_multiple_function=score_multiple_rmse
    ).run(df_pred_oof, ser_targets_train)
    print(final_weights)
    print(df_weights)

    ser_ensemble = np.matmul(df_pred_oof, final_weights)
    print(root_mean_squared_error(ser_targets_train, ser_ensemble))
