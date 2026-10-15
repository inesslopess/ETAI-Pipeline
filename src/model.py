"""Model construction."""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.random_projection import SparseRandomProjection

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
}

def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type not in _MODELS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Options: {list(_MODELS)}"
        )

    return _MODELS[model_type](**params)



def build_pipeline(preprocessing_config, model_config):

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

    steps.append((
        "model",
        build_model(model_config)
    ))

    return Pipeline(steps)
