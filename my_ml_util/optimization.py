import pandas as pd

from typing import Any


def get_weight_combinations(n_weights: int, step: int) -> list[tuple]:
    combinations = []

    def _get_weights(weight_options, current_weights):
        for w in weight_options:
            weights = current_weights + [w]
            if len(weights) == n_weights - 1:
                final_weights = weights + [100 - sum(weights)]
                combinations.append(tuple(final_weights))

            else:
                remaining = 100 - sum(weights)
                next_weight_options = range(0, remaining + 1, step)
                _get_weights(next_weight_options, current_weights + [w])
    weight_options = range(0, 101, step)
    _get_weights(weight_options, [])
    return combinations


def get_params_from_optuna_results(ser_results: pd.Series) -> dict[str, Any]:
    param_cols = [c for c in ser_results.index if c.startswith('params_')]
    ser_params = ser_results[param_cols]
    ser_params_dict = ser_params.to_dict()
    params = {k[7:]: v for k, v in ser_params_dict.items()}
    return params

