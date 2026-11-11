"""Evaluation with stratified cross-validation and out-of-fold probabilities."""

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    get_scorer,
    make_scorer,
)
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """
    {name: scorer} for scikit-learn metric names (`accuracy`, `roc_auc`, ...),
    plus `fbeta_<beta>` (e.g. `fbeta_2`: recall weighted twice as much as precision).
    """

    scorers = {}

    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(
                fbeta_score,
                beta=float(name.split("_", 1)[1]),
                zero_division=0,
            )
        else:
            scorers[name] = get_scorer(name)

    return scorers


def cross_validate_pipeline(
    pipeline,
    X,
    y,
    cv,
    scorers: dict,
    selection_metric: str,
    n_jobs: int = 1,
):
    """
    Fits a fresh copy of the whole pipeline on each fold.

    Because preprocessing is inside the pipeline, imputation,
    encoding and scaling are learned only from each fold's
    training data.

    Returns:
        fold_scores:
            one row per fold with the validation score of every
            metric, plus the train score and train-validation gap
            of `selection_metric`.

        oof_proba:
            out-of-fold probability of the positive class. Each row
            is predicted by a model that did not train on that row.
    """

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scorers,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=n_jobs,
    )

    fold_scores = pd.DataFrame({
        "fold": range(1, cv.get_n_splits() + 1),
        "train": scores[f"train_{selection_metric}"],
        "validation": scores[f"test_{selection_metric}"],
    })

    fold_scores["gap"] = (
        fold_scores["train"] - fold_scores["validation"]
    )

    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    # One probability column per class (2 for yes/no, 3 for Low/Medium/High),
    # in sklearn's order (sorted), named after the class.
    classes = np.unique(y)
    oof_proba = np.empty((len(X), len(classes)))

    for model, val_idx in zip(
        scores["estimator"],
        scores["indices"]["test"],
    ):
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])

    oof_proba = pd.DataFrame(oof_proba, index=X.index, columns=classes)

    return fold_scores, oof_proba
    


def cv_report(
    fold_scores: pd.DataFrame,
    metrics: list,
    selection_metric: str,
    title: str,
) -> str:
    """
    Returns (does not print):
    - result of every fold
    - mean and standard deviation of every metric
    - train-validation gap of the selection metric
    """

    lines = [
        f"{title} ({len(fold_scores)} stratified folds, "
        f"selection metric: {selection_metric})",
        "",
        fold_scores.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}",
        ),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(
            float_format=lambda v: f"{v:.3f}",
        ),
        f"\n{selection_metric}: train-validation gap "
        f"mean = {fold_scores['gap'].mean():+.3f}",
    ]

    return "\n".join(lines)


def predict_from_proba(oof_proba: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """
    One predicted class per row, from the probability table.
      - 2 classes: the second class if its probability >= threshold
        (the movable yes/no threshold, exactly as before);
      - 3+ classes: the class with the highest probability (argmax).
    """
    classes = oof_proba.columns

    if len(classes) == 2:
        return pd.Series(
            np.where(oof_proba[classes[1]] >= threshold, classes[1], classes[0]),
            index=oof_proba.index,
        )

    return oof_proba.idxmax(axis=1)


def oof_classification_report(y_true, oof_proba: pd.DataFrame, threshold: float = 0.5) -> str:
    """
    Confusion matrix and classification report of the
    out-of-fold predictions on the development set.
    Works for 2 classes (threshold) and 3+ classes (highest probability).
    """

    classes = list(oof_proba.columns)
    y_pred = predict_from_proba(oof_proba, threshold)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=classes),
        index=[f"true {c}" for c in classes],
        columns=[f"predicted {c}" for c in classes],
    )

    rule = f"threshold {threshold}" if len(classes) == 2 else "highest probability"

    return (
        f"Out-of-fold predictions, development set ({rule}):\n\n"
        f"{cm.to_string()}\n\n"
        f"{classification_report(y_true, y_pred, labels=classes, zero_division=0)}"
    )



def fairness_report(
    y_true,
    oof_proba: pd.DataFrame,
    extras: pd.DataFrame,
    sensitive_attr: str = "race",
    threshold: float = 0.5,
    positive_class=None,
    compare_compas: bool = True,
) -> str:
    """
    Simple fairness check using out-of-fold predictions.

    `positive_class` is the class that counts as "flagged"
    (binary: 1, the default = the last class; three classes: e.g. "High").

    For each group of `sensitive_attr`, the false positive rate is:
        among people NOT truly in `positive_class`,
        the share predicted to be in it.

    For:
        - our model
        - COMPAS's own score ("score_text" other than "Low"), if
          `compare_compas` (turn it off when score_text is the target)

    Rows with an unknown group are reported as "unknown".
    """

    if positive_class is None:
        positive_class = oof_proba.columns[-1]

    df = extras.copy()

    df[sensitive_attr] = (
        df[sensitive_attr].astype(object).fillna("unknown")
    )
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (
        predict_from_proba(oof_proba, threshold).values == positive_class
    ).astype(int)

    predictions = [("Our model", "y_pred_model")]

    if compare_compas:
        df["y_pred_compas"] = (
            df["score_text"] != "Low"
        ).astype(int)
        predictions.append(("COMPAS's own score", "y_pred_compas"))

    lines = [
        f"False positive rate by {sensitive_attr} "
        "(development set, out-of-fold)",
        f"(share of people NOT truly '{positive_class}', "
        f"but predicted '{positive_class}')",
        "",
    ]

    for label, col in predictions:

        lines.append(f"  {label}:")

        for group, g in df.groupby(sensitive_attr):

            negatives = g[g["y_true"] != positive_class]

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

    return "\n".join(lines)



def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models
    scored on the same CV folds.

    Folds share most of their training rows, so the plain t-test is
    overconfident; the variance of the differences is inflated by
    `test_train_ratio` (1 / (k - 1) for k-fold).

    Returns the mean difference (a - b), the folds where a scores
    higher, t and the two-sided p-value.
    """

    diff = (
        np.asarray(scores_a, dtype=float)
        - np.asarray(scores_b, dtype=float)
    )
    k = len(diff)

    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))

    return {
        "mean_diff": float(diff.mean()),
        "wins": f"{int((diff > 0).sum())}/{k}",
        "t": t,
        "p_value": p_value,
    }
