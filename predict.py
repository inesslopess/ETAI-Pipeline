"""
Predicts new people with the model saved by main.py (no training here).

Run with (from the project folder):
    python main.py                                                   # train, evaluate, save models/model.joblib
    python predict.py data/compas_new_people.csv predictions.csv     # predict

The new CSV is cleaned with the same rules as training, but WITHOUT
drop_duplicate_rows(): every input row gets an answer. Then it goes
through the saved Pipeline, with the values learned in training
(imputer, encoder, scaler, model). Nothing is re-learned on new data.

Safety checks (W10 challenge B4):
  - a column the model needs is missing -> stop with a clear message;
  - the libraries are not the versions the model was saved with -> warning.
"""
import argparse
import json
import os
import sys

import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, split_features_target
from src.results import library_versions, versions_path


def check_columns(df: pd.DataFrame, X: pd.DataFrame, pipeline, config: dict) -> None:
    """Raises a clear ValueError if the input lacks a column the model needs (or the id column)."""
    id_column = config["diagnostics"]["id_column"]
    needed = list(getattr(pipeline, "feature_names_in_", []))
    # `<col>_was_missing` flags are made from <col>, so report <col>, the column the user must add.
    missing = sorted({c.removesuffix("_was_missing") for c in needed if c not in X.columns})
    if id_column not in df.columns:
        missing.append(id_column)
    if missing:
        raise ValueError(f"the input is missing {len(missing)} column(s) the model needs: {', '.join(missing)}")


def check_versions(model_path: str) -> list:
    """Warnings for every library whose version differs from the one the model was saved with."""
    path = versions_path(model_path)
    if not os.path.exists(path):
        return [f"no {path}: can't check the library versions the model was saved with"]
    with open(path) as f:
        saved = json.load(f)
    now = library_versions()
    return [f"{name}: model saved with {version}, now {now.get(name)}"
            for name, version in saved.items() if now.get(name) != version]


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """Returns the id, the probability of the positive class and the prediction (at 0.5) for every row."""
    clean = clean_dataset(df, config["diagnostics"])
    X, _, _ = split_features_target(clean, config["data"], config["preprocessing"]["mnar_indicator_sources"])
    check_columns(df, X, pipeline, config)
    probability = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame({"id": df[config["diagnostics"]["id_column"]],
                         "probability": probability,
                         "prediction": (probability >= 0.5).astype(int)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_path = config["output"]["model_path"]
    if not os.path.exists(model_path):
        sys.exit(f"Error: no saved model at {model_path}. Run `python main.py` first.")

    for warning in check_versions(model_path):
        print(f"Warning: {warning}")

    pipeline = joblib.load(model_path)
    try:
        predictions = predict(pd.read_csv(args.input), pipeline, config)
    except ValueError as error:
        sys.exit(f"Error: {error}")
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")
