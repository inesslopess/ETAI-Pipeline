"""Model construction."""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.random_projection import SparseRandomProjection
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
import numpy as np

from sklearn.pipeline import Pipeline
from src.preprocessing import (
    build_preprocessor,
    add_engineered_features
)
from sklearn.preprocessing import FunctionTransformer


def oblique_random_forest(
    n_estimators=100,
    n_components=10,
    density="auto",
    random_state=None,
    n_jobs=None,
    **tree_params
):
    tree_params.setdefault("max_features", "sqrt")

    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(
            n_components=n_components,
            density=density
        )),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])

    return BaggingClassifier(
        estimator=oblique_tree,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )


def adaboost(max_depth=1, **params):
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        **params
    )


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
}

def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Options: {list(_MODELS)}"
        )

    return _MODELS[model_type](**params)



class VoteSelector(SelectorMixin, BaseEstimator):
    """
    Keeps the features chosen by at least `min_votes` of three models,
    fitted on the training rows only:
        - random forest: importance above the mean
        - gradient boosting: importance above the mean
        - L1 logistic regression: coefficient not shrunk to zero
    """

    def __init__(self, min_votes: int = 2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        forest = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X, y)
        booster = GradientBoostingClassifier(
            random_state=self.random_state
        ).fit(X, y)
        lasso = LogisticRegression(
            l1_ratio=1.0,
            C=0.01,
            solver="saga",
            max_iter=5000,
        ).fit(X, y)

        self.votes_ = (
            (forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
            + (booster.feature_importances_ > booster.feature_importances_.mean())
            + (np.abs(lasso.coef_[0]) > 1e-8)
        )
        self.support_ = self.votes_ >= self.min_votes
        return self

    def _get_support_mask(self):
        return self.support_


def build_pipeline(preprocessing_config, model_config, selection_config=None):
    """
    features (optional) -> prep -> select (optional) -> model.

    Every learned step is inside the Pipeline, so it is re-fitted
    on the training part of each fold.
    """

    steps = []

    feature_config = preprocessing_config.get(
        "feature_engineering", {}
    )

    if feature_config.get("enabled", False):
        steps.append((
            "features",
            FunctionTransformer(
                add_engineered_features,
                validate=False
            )
        ))

    steps.append((
        "prep",
        build_preprocessor(preprocessing_config)
    ))

    selection_config = selection_config or {}

    if selection_config.get("enabled", False):
        steps.append((
            "select",
            VoteSelector(
                min_votes=selection_config.get("min_votes", 2),
                random_state=selection_config.get("random_state"),
            )
        ))

    steps.append((
        "model",
        build_model(model_config)
    ))

    return Pipeline(steps)
