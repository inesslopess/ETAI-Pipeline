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
       (every metric in cv.metrics, out-of-fold probabilities)
    -> no-model floor (dummy) on the same folds
    -> optional tuning on cv.selection_metric (nested CV)
    -> out-of-fold evaluation (confusion matrix, fairness)
    -> refit final model on the full development set
    -> save the report and add one row to the scoreboard
"""

from datetime import datetime

import optuna
import yaml

from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    split_dev_test,
)
from src.model import build_pipeline
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report,
    corrected_paired_test,
)
from src.results import save_run, save_scores
from src.tuning import (
    tune_pipeline,
    nested_cross_validate,
    tuning_report,
)



def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]
    selection_config = config.get("feature_selection", {})
    selection_enabled = selection_config.get("enabled", False)

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

    # ---------------------------------------------------------
    # 6. METRICS AND STRATIFIED CROSS-VALIDATION
    # ---------------------------------------------------------
    # Every metric in cv.metrics is reported;
    # cv.selection_metric is the one tuning maximises
    # and runs are ranked by.
    cv_config = config["cv"]
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)

    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )

    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
          f"{cv_config['n_splits']}-fold CV"
          + ("   |   feature selection: on" if selection_enabled else ""))
    print(f"Data:  {len(X_dev)} development rows   |   "
          f"{len(X_test)} test rows locked away (not used)")

    # ---------------------------------------------------------
    # 7. BUILD THE WHOLE PIPELINE
    # ---------------------------------------------------------
    # Preprocessing (and feature selection, if enabled) MUST be
    # inside the sklearn Pipeline, so every CV fold learns its own
    # imputations, encoding, scaling and selected features
    # using only that fold's training data.
    pipeline = build_pipeline(
        config["preprocessing"],
        config["model"],
        selection_config,
    )

    # ---------------------------------------------------------
    # 8. CROSS-VALIDATE ON DEVELOPMENT SET (+ NO-MODEL FLOOR)
    # ---------------------------------------------------------
    floor_pipeline = build_pipeline(
        config["preprocessing"],
        {"type": "dummy"},
    )

    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline, X_dev, y_dev, cv,
        {selection_metric: scorers[selection_metric]},
        selection_metric, n_jobs,
    )

    fold_scores, oof_proba = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv,
        scorers, selection_metric, n_jobs,
    )

    floor = floor_scores["validation"].mean()
    untuned = fold_scores["validation"].mean()

    print("\nDefault hyperparameters (config.yaml)")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")

    report = (
        f"Hyperparameters from config.yaml: {config['model'].get('params')}\n\n"
        + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
        + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}"
    )

    final_scores = fold_scores

    # ---------------------------------------------------------
    # 9. HYPERPARAMETER TUNING (OPTIONAL)
    # ---------------------------------------------------------
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)

    if tuning_enabled:

        search_spaces = tuning_config.get("search_spaces") or {}

        if model_type not in search_spaces:
            raise ValueError(
                f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                f"'{model_type}'. Add one, or set tuning.enabled: false."
            )

        search_space = dict(search_spaces[model_type] or {})

        # Feature selection's own hyperparameters are tuned with the model.
        if selection_enabled:
            search_space.update(selection_config.get("search_space") or {})

        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed,
        )

        # 1. Honest estimate on the same outer folds;
        #    its out-of-fold probabilities feed the reports.
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)

        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv,
            scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )

        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")

        report += "\n\n" + cv_report(
            final_scores, metrics, selection_metric,
            "Nested cross-validation (the tuning procedure, estimated honestly)",
        )

        # 2. The hyperparameters that are kept:
        #    the same procedure, once, on all development rows.
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv,
            scorers[selection_metric],
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )

        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            print(f"    {name.replace('model__', ''):<20s} {value:.4g}" if isinstance(value, float)
                  else f"    {name.replace('model__', ''):<20s} {value}")

        report += "\n\n" + tuning_report(study, final_scores, selection_metric)

    # ---------------------------------------------------------
    # 10. EVERY METRIC, CONFUSION MATRIX, FAIRNESS
    # ---------------------------------------------------------
    print(f"\nAll metrics ({'tuned model' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")

    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(
        y_dev,
        oof_proba,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    # ---------------------------------------------------------
    # 11. REFIT FINAL MODEL ON ALL DEVELOPMENT DATA
    # ---------------------------------------------------------
    # X_test / y_test are still NOT scored here.
    pipeline.fit(X_dev, y_dev)

    refit = f"Final model: {model_type} refit on all {len(X_dev)} development rows."
    report += "\n" + refit + "\n"

    # ---------------------------------------------------------
    # 12. SCOREBOARD: ONE ROW PER RUN, ALL RUNS IN ONE FILE
    # ---------------------------------------------------------
    # Only runs on the same data, cleaning and folds are comparable.
    # If any of those change, delete results/runs.csv and start again.
    row = {
        "run": run,
        "data": config["data"]["path"],
        "model": model_type,
        "tuned": tuning_enabled,
        "feature_selection": selection_enabled,
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "params": str(config["model"].get("params")),
        "tuned_params": str(study.best_params) if tuning_enabled else "",
        "metric": selection_metric,
        "mean": final_scores["validation"].mean(),
        "std": final_scores["validation"].std(ddof=1),
        "untuned": untuned,
        "floor": floor,
        **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)},
        **{f"{name}_mean": final_scores[name].mean() for name in metrics},
    }

    runs = save_scores(
        config.get("output", {}).get("scoreboard", "results/runs.csv"),
        row,
    )

    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
    same = same.sort_values("mean", ascending=False)

    board = (
        same[["run", "model", "tuned", "feature_selection", "encoder", "scaler",
              "mean", "std", "untuned", "floor"]]
        .rename(columns={"mean": selection_metric})
        .to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    # This run against the best other run, on the same folds.
    others = same[same["run"] != run]
    if len(others) > 0:
        best = others.iloc[0]
        folds = [f"fold_{i}" for i in range(1, cv_config["n_splits"] + 1)]
        test = corrected_paired_test(
            final_scores["validation"],
            best[folds].astype(float),
            test_train_ratio=1 / (cv_config["n_splits"] - 1),
        )
        comparison = (
            f"This run vs best other run ({best['model']}"
            f"{', tuned' if best['tuned'] else ''}, run {best['run']}), corrected paired t-test:\n"
            f"  mean diff = {test['mean_diff']:+.4f}   wins = {test['wins']}   "
            f"t = {test['t']:.2f}   p = {test['p_value']:.3f}"
        )
        print(f"\n{comparison}")
        report += f"\n{comparison}\n"

    # ---------------------------------------------------------
    # 13. SAVE RESULTS
    # ---------------------------------------------------------
    path = save_run(
        config.get("output", {}).get("results_dir", "results"),
        run,
        config,
        report,
    )

    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")


if __name__ == "__main__":
    main()
