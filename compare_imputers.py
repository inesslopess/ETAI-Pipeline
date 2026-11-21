"""
W10 challenge B2: does the imputer guess better when it also sees the categorical columns?

We can't check a guess for a real gap, so we MAKE gaps where we know the answer
(as in 10_missing_values.ipynb, section 5): hide 200 known values of `age` and of
`priors_count` in the validation rows of fold 1, fit each imputer on the training
rows, and measure how far each guess lands from the truth (mean absolute error,
in years and in priors). Lower is better.

Run with:
    python compare_imputers.py
"""
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    _NUMERIC_IMPUTERS,
    _SCALERS,
    NumericImputerWithHelpers,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)

HIDDEN_COLUMNS = ["age", "priors_count"]
N_HIDDEN = 200


def load_dev(config):
    """The same development rows and fold 1 as main.py."""
    df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    df = drop_duplicate_rows(df, id_column=config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(df, config["data"], config["preprocessing"]["mnar_indicator_sources"])
    X_dev, _, y_dev, _, _, _ = split_dev_test(X, y, extras, config["test_set"]["size"], config["test_set"]["random_state"])
    cv = config["cv"]
    folds = StratifiedKFold(cv["n_splits"], shuffle=True, random_state=cv["random_state"])
    train_idx, val_idx = next(folds.split(X_dev, y_dev))
    return X_dev.iloc[train_idx], X_dev.iloc[val_idx]


def hide_values(X, rng):
    """Hides N_HIDDEN known values per column; returns the data with the new gaps and the hidden truth."""
    X = X.copy()
    truth = {}
    for column in HIDDEN_COLUMNS:
        hidden = rng.choice(X.index[X[column].notna()], N_HIDDEN, replace=False)
        truth[column] = X.loc[hidden, column].astype(float).copy()
        X.loc[hidden, column] = np.nan
    return X, truth


def guesses(strategy, use_categorical, X_train, X_val, numeric, categorical, prep):
    """The imputer's filled-in numeric columns for X_val, back in original units."""
    scaler = _SCALERS[prep["scaler"]]
    scaler = scaler() if callable(scaler) else scaler
    imputer = _NUMERIC_IMPUTERS[strategy](prep.get("imputation", {}), prep.get("random_state"))
    if use_categorical:
        step = NumericImputerWithHelpers(numeric, categorical, scaler, imputer).fit(X_train)
        filled, fitted_scaler = step.transform(X_val), step.scaler_
    else:
        step = Pipeline([("scale", clone(scaler)), ("impute", imputer)]).fit(X_train[numeric])
        filled, fitted_scaler = step.transform(X_val[numeric]), step["scale"]
    if fitted_scaler is not None and fitted_scaler != "passthrough":
        filled = fitted_scaler.inverse_transform(filled)
    return pd.DataFrame(filled, index=X_val.index, columns=numeric)


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    prep = config["preprocessing"]
    numeric, categorical = list(prep["numeric_features"]), list(prep["categorical_features"])

    X_train, X_val = load_dev(config)
    X_val_hidden, truth = hide_values(X_val, np.random.default_rng(0))

    rows = []
    for strategy in ["median", "knn", "iterative", "random_forest"]:
        for use_categorical in [False, True]:
            filled = guesses(strategy, use_categorical, X_train, X_val_hidden, numeric, categorical, prep)
            row = {"imputer": strategy, "sees categories": "yes" if use_categorical else "no"}
            for column, values in truth.items():
                row[f"{column} error"] = (filled.loc[values.index, column] - values).abs().mean()
            rows.append(row)
            print(f"  done: {strategy:<14s} sees categories: {row['sees categories']}", flush=True)

    print(f"\nMean absolute error of the guesses ({N_HIDDEN} hidden values per column, fold 1; lower is better)")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
