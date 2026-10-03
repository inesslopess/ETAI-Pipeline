"""Evaluation with stratified cross-validation and out-of-fold predictions."""

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate


def cross_validate_pipeline(
    pipeline,
    X,
    y,
    cv,
    scoring: str = "accuracy",
    n_jobs: int = 1,
):
    """
    Fits a fresh copy of the whole pipeline on each fold.

    Because preprocessing is inside the pipeline, imputation,
    encoding and scaling are learned only from each fold's
    training data.

    Returns:
        fold_scores:
            one row per fold with train score, validation score
            and train-validation gap.

        y_oof:
            out-of-fold predictions. Each row is predicted by
            a model that did not train on that row.
    """

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=n_jobs,
    )

    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })

    fold_scores["gap"] = (
        fold_scores["train"] - fold_scores["validation"]
    )

    y_oof = np.empty(
        len(X),
        dtype=np.asarray(y).dtype,
    )

    for model, val_idx in zip(
        scores["estimator"],
        scores["indices"]["test"],
    ):
        y_oof[val_idx] = model.predict(X.iloc[val_idx])

    return fold_scores, y_oof


def cv_report(
    fold_scores: pd.DataFrame,
    scoring: str = "accuracy",
) -> str:
    """
    Prints and returns:
    - result of every fold
    - mean
    - standard deviation
    - train-validation gap
    """

    lines = [
        f"Cross-validation ({len(fold_scores)} stratified folds, metric: {scoring})",
        "",
        fold_scores.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}",
        ),
        "",
    ]

    for col in ["train", "validation", "gap"]:
        sign = "+" if col == "gap" else ""

        lines.append(
            f"{col.capitalize():<11s} "
            f"mean = {fold_scores[col].mean():{sign}.3f}   "
            f"std = {fold_scores[col].std(ddof=1):.3f}"
        )

    text = "\n".join(lines)

    print(text)

    return text


def oof_classification_report(y_true, y_pred) -> str:
    """
    Classification report using out-of-fold predictions
    from the development set.
    """

    text = (
        "Classification report "
        "(out-of-fold predictions, development set):\n"
        + classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    print(text)

    return text


def fairness_report(
    y_true,
    y_pred,
    extras: pd.DataFrame,
    sensitive_attr: str = "race",
) -> str:
    """
    Simple fairness check using out-of-fold predictions.

    For each race group, calculates the false positive rate
    for:
        - our model
        - COMPAS's own score
    """

    df = extras.copy()

    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = y_pred

    df["y_pred_compas"] = (
        df["score_text"] != "Low"
    ).astype(int)

    lines = [
        "False positive rate by race "
        "(development set, out-of-fold)",
        "(share of people who did NOT reoffend, "
        "but were predicted to)",
        "",
    ]

    for label, col in [
        ("Our model", "y_pred_model"),
        ("COMPAS's own score", "y_pred_compas"),
    ]:

        lines.append(f"  {label}:")

        for group, g in df.groupby(sensitive_attr):

            negatives = g[g["y_true"] == 0]

            if len(negatives) == 0:
                continue

            fpr = (
                negatives[col] == 1
            ).mean()

            lines.append(
                f"    {group:<20s} "
                f"FPR = {fpr:.2f}  "
                f"(n={len(negatives)})"
            )

        lines.append("")

    text = "\n".join(lines)

    print(text)

    return text