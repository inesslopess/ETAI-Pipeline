"""
W11: are the people who reoffend the outliers?

On the development rows (the same as main.py), the features + preprocessor are
fitted once and each detector scores every person (as in 12_outliers.ipynb).
Then: how often do the most unusual 5% reoffend, compared with everyone else?
This only describes the data (no model, no CV): main.py measures whether the
score helps a model.

Run with:
    python check_anomalies.py
"""
import numpy as np
import pandas as pd
import yaml

from src.data import load_data
from src.model import build_pipeline
from src.preprocessing import (
    AnomalyScore,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)

TOP_SHARE = 0.05      # the most unusual 5%


def load_dev(config):
    """The same development rows as main.py."""
    df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    df = drop_duplicate_rows(df, id_column=config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(df, config["data"], config["preprocessing"]["mnar_indicator_sources"])
    X_dev, _, y_dev, _, _, _ = split_dev_test(X, y, extras, config["test_set"]["size"], config["test_set"]["random_state"])
    return X_dev, y_dev.astype(int).to_numpy()


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    X_dev, y_dev = load_dev(config)

    # Everything before the model (features + prep), fitted on the dev rows.
    before_model = build_pipeline(config["preprocessing"], {"type": "dummy"})[:-1]
    Z = before_model.fit_transform(X_dev, y_dev)

    numeric = config["preprocessing"]["numeric_features"]
    print(f"{len(y_dev)} development people, {y_dev.mean():.1%} reoffend within two years\n")

    for method in ["isolation_forest", "lof"]:
        score = AnomalyScore(method=method, random_state=42).fit_transform(Z)[:, -1]
        top = score >= np.quantile(score, 1 - TOP_SHARE)
        rho = pd.Series(score).corr(pd.Series(y_dev), method="spearman")

        print(f"{method}")
        print(f"  top {TOP_SHARE:.0%} most unusual ({top.sum()} people): {y_dev[top].mean():.1%} reoffend")
        print(f"  everyone else       ({(~top).sum()} people): {y_dev[~top].mean():.1%} reoffend")
        print(f"  rank correlation of score with reoffending: {rho:+.2f}")
        print("  typical top person (median):  "
              + ", ".join(f"{c} {X_dev.loc[top, c].median():g}" for c in numeric))
        print("  typical other person (median): "
              + ", ".join(f"{c} {X_dev.loc[~top, c].median():g}" for c in numeric) + "\n")


if __name__ == "__main__":
    main()
