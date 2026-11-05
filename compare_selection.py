"""
With vs without feature selection, model by model.

Run with:
    python compare_selection.py

For every model on the scoreboard (results/runs.csv) that has a TUNED run
both with and without feature selection, this takes the most recent run of
each and compares them with the corrected paired t-test, fold by fold.
Both runs use the same folds (same cv.random_state), so the test is valid.

Reads only the scoreboard: nothing is trained, and the test set is not used.
"""

import pandas as pd
import yaml

from src.evaluate import corrected_paired_test


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    k = config["cv"]["n_splits"]
    metric = config["cv"]["selection_metric"]
    folds = [f"fold_{i}" for i in range(1, k + 1)]

    runs = pd.read_csv(config["output"]["scoreboard"], dtype={"run": str})
    runs = runs[(runs["tuned"]) & (runs["metric"] == metric)]

    rows = []
    for model, group in runs.groupby("model"):
        with_sel = group[group["feature_selection"]].sort_values("run")
        without = group[~group["feature_selection"]].sort_values("run")
        if with_sel.empty or without.empty:
            continue

        a, b = with_sel.iloc[-1], without.iloc[-1]   # most recent run of each
        test = corrected_paired_test(
            a[folds].astype(float),
            b[folds].astype(float),
            test_train_ratio=1 / (k - 1),
        )
        rows.append({
            "model": model,
            "without": f"{b['mean']:.3f} ± {b['std']:.3f}",
            "with selection": f"{a['mean']:.3f} ± {a['std']:.3f}",
            "diff (with - without)": round(test["mean_diff"], 4),
            "wins": test["wins"],
            "t": round(test["t"], 2),
            "p": round(test["p_value"], 3),
        })

    if not rows:
        print("No model has tuned runs both with and without feature selection yet.")
        return

    table = pd.DataFrame(rows).sort_values("diff (with - without)", ascending=False)
    print(f"Tuned runs, {metric}, {k}-fold CV, corrected paired t-test\n")
    print(table.to_string(index=False))
    print(f"\n{len(rows)} tests: with this many, one p < 0.05 can appear by chance.")


if __name__ == "__main__":
    main()
