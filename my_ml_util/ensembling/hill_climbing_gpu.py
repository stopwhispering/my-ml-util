from sklearn.metrics import root_mean_squared_error

from podcast.constants import PATH_TRAIN, PATH_INTERIM_RESULTS
import pandas as pd
import numpy as np
import cupy as cp
import gc

df_train = (
    pd.read_csv(PATH_TRAIN).set_index("id").drop("Listening_Time_minutes", axis=1)
)
ser_targets_train = pd.read_csv(PATH_TRAIN).set_index("id")["Listening_Time_minutes"]

ser_pred_lgbm = pd.read_pickle(PATH_INTERIM_RESULTS / "ser_oof_preds_lgbm.pkl")
ser_pred_xgb = pd.read_pickle(PATH_INTERIM_RESULTS / "ser_oof_preds_xgb.pkl")
df_pred_oof = pd.concat([ser_pred_lgbm, ser_pred_xgb], axis=1)
df_pred_oof.columns = ["lgbm", "xgb"]


# compute metric for each oof
best_score = 1000
best_index = -1

for col_idx, col in enumerate(df_pred_oof.columns):
    # score = roc_auc_score(ser_targets_train, df_pred_oof[col])
    score = root_mean_squared_error(ser_targets_train, df_pred_oof[col])
    if score < best_score:
        best_score = score
        best_index = col_idx
    print(f"Score {score:0.5f} {col}")
print()
print(
    f"Best single model is {df_pred_oof.columns[best_index]} with Score = {best_score:0.5f}"
)


def multiple_scores(actual, predicted):
    """ """
    scores = []
    for i in range(predicted.shape[1]):
        score = root_mean_squared_error(actual, predicted[:, i])
        scores.append(score)
        # print(score)
    return np.array(scores)

    # n_pos = cp.sum(actual)  # Number of positive samples (on GPU)
    # n_neg = len(actual) - n_pos  # Number of negative samples (on GPU)
    # ranked = cp.argsort(cp.argsort(predicted, axis=0), axis=0) + 1  # Ranks for each column (on GPU)
    # aucs = (cp.sum(ranked[actual == 1, :], axis=0) - n_pos * (n_pos + 1) / 2) / (
    #             n_pos * n_neg)  # AUC computation
    # return aucs  # AUC scores for each classifier (on GPU)


# cdeotte implementation from https://www.kaggle.com/code/cdeotte/fast-gpu-hill-climbing-starter-cv-0-94-lb-0-94

USE_NEGATIVE_WGT = True
MAX_MODELS = 1000
TOL = 1e-6

indices = [best_index]
old_best_score = best_score

# PREPARE/MOVE VARIABLES TO GPU FOR SPEED UP
# x_train2 = cp.array( np.log( df_pred_oof/(1-df_pred_oof) ) ) #GPU LOGITS
x_train2 = cp.array(df_pred_oof)  # GPU LOGITS
best_ensemble = x_train2[:, best_index]  # GPU
truth = cp.array(ser_targets_train.values)  # GPU
start = -0.50
if not USE_NEGATIVE_WGT:
    start = 0.01
ww = cp.arange(start, 0.51, 0.01)  # GPU
nn = len(ww)

# BEGIN HILL CLIMBING
models = [best_index]
weights = []
metrics = [best_score]

for kk in range(10_000_000):
    best_score = 0
    best_index = -1
    best_weight = 0

    # TRY ADDING ONE MORE MODEL
    for k, ff in enumerate(df_pred_oof.columns):
        new_model = x_train2[:, k]  # GPU
        m1 = cp.repeat(best_ensemble[:, cp.newaxis], nn, axis=1) * (1 - ww)  # GPU
        m2 = cp.repeat(new_model[:, cp.newaxis], nn, axis=1) * ww  # GPU
        mm = m1 + m2  # GPU
        # mm = 1 / (1 + cp.exp(-mm)) # GPU (convert logits to probs - not needed for auc)
        new_aucs = multiple_roc_auc_scores(truth, mm)
        new_score = cp.max(new_aucs).item()  # GPU -> CPU
        if new_score > best_score:
            best_score = new_score  # CPU
            best_index = k  # CPU
            ii = np.argmax(new_aucs).item()  # GPU -> CPU
            best_weight = ww[ii].item()  # GPU -> CPU
            potential_ensemble = mm[:, ii]  # GPU
    # del new_model, m1, m2, mm, new_aucs, new_score  #todo
    gc.collect()

    # STOPPING CRITERIA
    indices.append(best_index)
    indices = list(np.unique(indices))
    if len(indices) > MAX_MODELS:
        print(f"=> We reached {MAX_MODELS} models")
        indices = indices[:-1]
        break
    if best_score - old_best_score < TOL:
        print(f"=> We reached tolerance {TOL}")
        break

    # RECORD NEW RESULT
    print(
        kk,
        "New best AUC",
        best_score,
        f'adding "{df_pred_oof.columns[best_index]}"',
        "with weight",
        f"{best_weight:0.3f}.",
    )
    models.append(best_index)
    weights.append(best_weight)
    metrics.append(best_score)
    best_ensemble = potential_ensemble
    old_best_score = best_score

print(f"{kk=}")
