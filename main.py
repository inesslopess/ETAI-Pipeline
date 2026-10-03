"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config
    -> load data
    -> clean data
    -> remove duplicate training rows
    -> split features/target
    -> lock final test set
    -> stratified 5-fold cross-validation on development data
    -> out-of-fold evaluation
    -> refit final model on the full development set
    -> save results
"""

import yaml

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    build_preprocessor,
    split_dev_test,
)
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    df_raw = load_data(config["data"]["path"])

    # ---------------------------------------------------------
    # 2. CLEAN DATA
    # ---------------------------------------------------------
    # General cleaning only:
    # placeholders -> NaN, category cleanup, invalid values,
    # redundant-column removal.
    #
    # clean_dataset() does NOT remove rows anymore.
    df_clean = clean_dataset(
        df_raw,
        config["diagnostics"],
    )

    # ---------------------------------------------------------
    # 3. REMOVE DUPLICATES
    # ---------------------------------------------------------
    # Duplicate removal is training-dataset preparation.
    # It is kept separate from general cleaning.
    df_clean = drop_duplicate_rows(
        df_clean,
        id_column=config["diagnostics"].get("id_column"),
    )

    # ---------------------------------------------------------
    # 4. SPLIT FEATURES / TARGET / EXTRAS
    # ---------------------------------------------------------
    mnar_sources = config["preprocessing"].get(
        "mnar_indicator_sources",
        [],
    )

    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        mnar_sources,
    )

    # ---------------------------------------------------------
    # 5. LOCK FINAL TEST SET
    # ---------------------------------------------------------
    # The 20% test set is created once and remains untouched.
    #
    # Cross-validation happens ONLY on X_dev / y_dev.
    (
        X_dev,
        X_test,
        y_dev,
        y_test,
        extras_dev,
        extras_test,
    ) = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    print(f"Development set size: {len(X_dev)}")
    print(f"Locked test set size: {len(X_test)}")
    print()

    # ---------------------------------------------------------
    # 6. BUILD THE WHOLE PIPELINE
    # ---------------------------------------------------------
    # Preprocessing MUST be inside the sklearn Pipeline.
    #
    # This means every CV fold learns its own:
    # - imputations
    # - target encoding
    # - scaling
    #
    # using only that fold's training data.
    pipeline = Pipeline([
        (
            "prep",
            build_preprocessor(config["preprocessing"]),
        ),
        (
            "model",
            build_model(config["model"]),
        ),
    ])

    # ---------------------------------------------------------
    # 7. DEFINE STRATIFIED CROSS-VALIDATION
    # ---------------------------------------------------------
    cv = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        shuffle=config["cv"]["shuffle"],
        random_state=config["cv"]["random_state"],
    )

    # ---------------------------------------------------------
    # 8. CROSS-VALIDATE ON DEVELOPMENT SET
    # ---------------------------------------------------------
    fold_scores, y_oof = cross_validate_pipeline(
        pipeline=pipeline,
        X=X_dev,
        y=y_dev,
        cv=cv,
        scoring=config["cv"]["scoring"],
        n_jobs=config["cv"]["n_jobs"],
    )

    # ---------------------------------------------------------
    # 9. CREATE REPORTS USING CV / OOF RESULTS
    # ---------------------------------------------------------
    report = cv_report(
        fold_scores,
        scoring=config["cv"]["scoring"],
    )

    report += "\n\n" + oof_classification_report(
        y_dev,
        y_oof,
    )

    report += "\n\n" + fairness_report(
        y_dev,
        y_oof,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    # ---------------------------------------------------------
    # 10. REFIT FINAL MODEL ON ALL DEVELOPMENT DATA
    # ---------------------------------------------------------
    # CV models were temporary evaluation models.
    #
    # Now we fit one fresh final pipeline using ALL development data.
    #
    # IMPORTANT:
    # X_test / y_test are still NOT scored here.
    final_pipeline = Pipeline([
        (
            "prep",
            build_preprocessor(config["preprocessing"]),
        ),
        (
            "model",
            build_model(config["model"]),
        ),
    ])

    final_pipeline.fit(
        X_dev,
        y_dev,
    )

    print()
    print("Final pipeline fitted on the full development set.")
    print("Locked test set has not been scored.")

    # ---------------------------------------------------------
    # 11. SAVE RESULTS
    # ---------------------------------------------------------
    results_dir = config.get(
        "output",
        {},
    ).get(
        "results_dir",
        "results",
    )

    path = save_run(
        results_dir,
        config,
        report,
    )

    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()