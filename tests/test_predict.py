"""
W10 challenge B4: small tests for predict.py.

Self-contained: trains a quick logistic regression (seconds, no tuning), saves it
with the same save_model() main.py uses, and checks predict.py with it. So it
doesn't depend on (or overwrite) models/model.joblib.

Run with (from the project folder):
    python tests/test_predict.py
"""
import os
import sys
import tempfile

import joblib
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the project folder

from predict import check_versions, predict
from src.data import load_data
from src.model import build_pipeline
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target
from src.results import save_model, versions_path

NEW_PEOPLE = "data/compas_new_people.csv"


def saved_test_model(config, folder):
    """A quick model, trained like main.py does (cleaning, duplicates removed) and saved."""
    df = drop_duplicate_rows(clean_dataset(load_data(config["data"]["path"]), config["diagnostics"]),
                             id_column=config["diagnostics"]["id_column"])
    X, y, _ = split_features_target(df, config["data"], config["preprocessing"]["mnar_indicator_sources"])
    pipeline = build_pipeline(config["preprocessing"], {"type": "logistic_regression", "params": {"max_iter": 2000}})
    model_path = os.path.join(folder, "model.joblib")
    save_model(pipeline.fit(X, y), model_path)
    return model_path


def test_rows_out_equal_rows_in(pipeline, config):
    new_people = pd.read_csv(NEW_PEOPLE)
    predictions = predict(new_people, pipeline, config)
    assert len(predictions) == len(new_people), f"{len(new_people)} rows in, {len(predictions)} out"
    assert list(predictions["id"]) == list(new_people["id"]), "the ids are not in the input order"
    assert predictions["probability"].between(0, 1).all() and predictions["probability"].notna().all()


def test_duplicates_are_kept(pipeline, config):
    new_people = pd.read_csv(NEW_PEOPLE)
    doubled = pd.concat([new_people, new_people.iloc[[0]]], ignore_index=True)    # the same person twice
    predictions = predict(doubled, pipeline, config)
    assert len(predictions) == len(doubled)
    # same person -> same answer (up to tiny rounding differences of the computer)
    first, copy = predictions["probability"].iloc[0], predictions["probability"].iloc[-1]
    assert abs(first - copy) < 1e-9, f"same person, different probabilities: {first} vs {copy}"


def test_missing_column_gives_clear_error(pipeline, config):
    without_priors = pd.read_csv(NEW_PEOPLE).drop(columns=["priors_count"])
    try:
        predict(without_priors, pipeline, config)
    except ValueError as error:
        assert "priors_count" in str(error), f"the message doesn't name the column: {error}"
    else:
        raise AssertionError("no error for a missing column")


def test_versions_saved_next_to_model(model_path):
    assert os.path.exists(versions_path(model_path)), "no versions file next to the model"
    assert check_versions(model_path) == [], "same environment, but the versions don't match"


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    with tempfile.TemporaryDirectory() as folder:
        model_path = saved_test_model(config, folder)
        pipeline = joblib.load(model_path)
        tests = [
            ("rows out = rows in", lambda: test_rows_out_equal_rows_in(pipeline, config)),
            ("duplicates are kept", lambda: test_duplicates_are_kept(pipeline, config)),
            ("missing column -> clear error", lambda: test_missing_column_gives_clear_error(pipeline, config)),
            ("library versions saved", lambda: test_versions_saved_next_to_model(model_path)),
        ]
        failed = 0
        for name, test in tests:
            try:
                test()
                print(f"  PASS  {name}")
            except AssertionError as error:
                failed += 1
                print(f"  FAIL  {name}: {error}")

    print(f"\n{len(tests) - failed} of {len(tests)} tests passed")
    sys.exit(1 if failed else 0)
