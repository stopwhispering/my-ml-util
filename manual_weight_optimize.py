
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-07T10:56:47.564119Z","iopub.execute_input":"2024-04-07T10:56:47.564601Z","iopub.status.idle":"2024-04-07T10:56:49.606200Z","shell.execute_reply.started":"2024-04-07T10:56:47.564559Z","shell.execute_reply":"2024-04-07T10:56:49.604473Z"}}
from my_ml_util.optimization import get_weight_combinations
import pandas as pd
import numpy as np
from sklearn import linear_model, svm, tree, neighbors, ensemble

# %% [code] {"execution":{"iopub.status.busy":"2024-04-07T10:56:49.608231Z","iopub.execute_input":"2024-04-07T10:56:49.608814Z","iopub.status.idle":"2024-04-07T10:56:49.627445Z","shell.execute_reply.started":"2024-04-07T10:56:49.608759Z","shell.execute_reply":"2024-04-07T10:56:49.625751Z"}}
samples = (
    (0.28, 0.18, 0.54, 0.00, 0.00),
    (0.00, 0.00, 0.00, 0.10, 0.00),
    (0.24, 0.14, 0.50, 0.12, 0.00),
    (0.20, 0.10, 0.46, 0.24, 0.00),
    (0.00, 0.00, 0.00, 0.00, 0.10),
    (0.24, 0.14, 0.50, 0.00, 0.12),
    (0.20, 0.10, 0.46, 0.00, 0.24),
    (0.10, 0.10, 0.40, 0.25, 0.15),
    (0.06, 0.06, 0.44, 0.38, 0.06),
    (0.05, 0.05, 0.50, 0.38, 0.02),
    (0.30, 0.00, 0.50, 0.20, 0.00),
    (0.00, 0.30, 0.50, 0.20, 0.00),
    (0.30, 0.00, 0.50, 0.10, 0.10),
)

targets = (
    0.29,
    0.31,
    0.29,
    0.28,
    0.36,
    0.29,
    0.29,
    0.29,
    0.28,
    0.29,
    0.28,
    0.30,
    0.29,
)

assert len(samples) == len(targets)
df_x = pd.DataFrame(samples)
df_x.columns = [str(c) for c in df_x.columns]
ser_y = pd.Series(targets)

print(f'{df_x.shape=}')
print(f'{ser_y.shape=}')

# %% [code] {"execution":{"iopub.status.busy":"2024-04-07T10:56:49.631442Z","iopub.execute_input":"2024-04-07T10:56:49.632274Z","iopub.status.idle":"2024-04-07T10:56:49.650113Z","shell.execute_reply.started":"2024-04-07T10:56:49.632220Z","shell.execute_reply":"2024-04-07T10:56:49.648330Z"}}
df_x['n_models'] = (df_x > 0).sum(axis=1)
print(f'{df_x.shape=}')

# %% [code] {"execution":{"iopub.status.busy":"2024-04-07T10:56:49.652910Z","iopub.execute_input":"2024-04-07T10:56:49.653449Z","iopub.status.idle":"2024-04-07T10:56:49.683158Z","shell.execute_reply.started":"2024-04-07T10:56:49.653373Z","shell.execute_reply":"2024-04-07T10:56:49.682033Z"}}
df_x

# %% [code] {"execution":{"iopub.status.busy":"2024-04-07T10:56:49.684336Z","iopub.execute_input":"2024-04-07T10:56:49.684758Z","iopub.status.idle":"2024-04-07T10:56:50.936971Z","shell.execute_reply.started":"2024-04-07T10:56:49.684725Z","shell.execute_reply":"2024-04-07T10:56:50.935501Z"}}
combinations = get_weight_combinations(5, 2)
print(f'{len(combinations)=:_}')

# combinations = tuple(c for c in combinations if min(c) >= 2)
# print(f'{len(combinations)=:_}')

df_pred = pd.DataFrame(combinations)
df_pred.columns = [str(c) for c in df_pred.columns]
df_pred /= 100
print(f'{df_pred.shape=}')

# %% [code] {"execution":{"iopub.status.busy":"2024-04-07T10:56:50.938929Z","iopub.execute_input":"2024-04-07T10:56:50.939353Z","iopub.status.idle":"2024-04-07T10:56:51.042129Z","shell.execute_reply.started":"2024-04-07T10:56:50.939316Z","shell.execute_reply":"2024-04-07T10:56:51.040762Z"}}
df_pred['n_models'] = (df_pred > 0).sum(axis=1)
df_pred

# %% [code] {"execution":{"iopub.status.busy":"2024-04-07T10:56:51.043960Z","iopub.execute_input":"2024-04-07T10:56:51.044520Z","iopub.status.idle":"2024-04-07T10:56:51.050789Z","shell.execute_reply.started":"2024-04-07T10:56:51.044472Z","shell.execute_reply":"2024-04-07T10:56:51.049366Z"}}
# static prediction

# svr = svm.SVR()
# svr.fit(df.iloc[:, :5],
#         df["target"])
# df_pred['svr'] = svr.predict(df_pred.iloc[:, :5])

# lasso = linear_model.Lasso(alpha=0.1)
# lasso.fit(df.iloc[:, :5],
#           df["target"])
# df_pred['lasso'] = lasso.predict(df_pred.iloc[:, :5])

# %% [code] {"execution":{"iopub.status.busy":"2024-04-07T10:56:51.052464Z","iopub.execute_input":"2024-04-07T10:56:51.052941Z","iopub.status.idle":"2024-04-07T10:56:51.802103Z","shell.execute_reply.started":"2024-04-07T10:56:51.052904Z","shell.execute_reply":"2024-04-07T10:56:51.800570Z"}}
decision_tree = tree.DecisionTreeRegressor()
decision_tree.fit(df_x,
                  ser_y)
df_pred['tree'] = decision_tree.predict(df_pred.iloc[:, :6])

knn = neighbors.KNeighborsRegressor(n_neighbors=2)
knn.fit(df_x,
        ser_y)
df_pred['knn'] = knn.predict(df_pred.iloc[:, :6])

random_forest = ensemble.RandomForestRegressor()
random_forest.fit(df_x,
                  ser_y)
df_pred['random_forest'] = random_forest.predict(df_pred.iloc[:, :6])

# %% [code] {"execution":{"iopub.status.busy":"2024-04-07T10:56:51.804091Z","iopub.execute_input":"2024-04-07T10:56:51.804583Z","iopub.status.idle":"2024-04-07T10:56:52.039136Z","shell.execute_reply.started":"2024-04-07T10:56:51.804545Z","shell.execute_reply":"2024-04-07T10:56:52.037528Z"}}
df_pred['mean'] = df_pred.iloc[:, 6:].mean(axis=1)
df_pred = df_pred.sort_values('mean', ascending=True)

for col in df_pred.columns[6:]:
    print(f'{col= :<20} ', f'mean = {df_pred[col].mean()}', f'std = {df_pred[col].std()}')


# %% [code]


# %% [markdown]
# # Conclusion
# The Models don't find any pattern. Probably weight optimization is just overfitting to public LB.

# %% [code] {"execution":{"iopub.status.busy":"2024-04-07T10:57:39.336987Z","iopub.execute_input":"2024-04-07T10:57:39.337521Z","iopub.status.idle":"2024-04-07T10:57:44.085584Z","shell.execute_reply.started":"2024-04-07T10:57:39.337483Z","shell.execute_reply":"2024-04-07T10:57:44.083741Z"}}
df_pred['str'] = (df_pred.iloc[:, :5]).round(2).astype(str).agg(', '.join, axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-07T10:57:46.829574Z","iopub.execute_input":"2024-04-07T10:57:46.830037Z","iopub.status.idle":"2024-04-07T10:57:46.859477Z","shell.execute_reply.started":"2024-04-07T10:57:46.830006Z","shell.execute_reply":"2024-04-07T10:57:46.858132Z"}}
df_pred

# %% [code]
