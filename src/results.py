"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes:
- the full report of each run (metrics, confusion matrix, fairness table, tuning) to a timestamped file in `results/`;
- one row per run to a single scoreboard CSV (`results/runs.csv`), so every run can be compared side by side.
"""
import os

import pandas as pd


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to `results_dir/run_<run>.txt`
    (the folder is created if needed) and returns the path.
    """
    os.makedirs(results_dir, exist_ok=True)

    path = os.path.join(results_dir, f"run_{run}.txt")
    cv_config = config["cv"]

    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Test size: {config['test_set']['size']}  "
        f"random_state: {config['test_set']['random_state']}\n"
        f"Metrics: {', '.join(cv_config['metrics'])}\n"
        f"Selection metric: {cv_config['selection_metric']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return path


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)

    return runs
