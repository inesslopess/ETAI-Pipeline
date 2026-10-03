"""
Preprocessing -- same file, same job as week 2 (raw data in, model-ready train/test
split out), now carrying the diagnosis-driven, leak-safe recipe this week's EDA
notebook justifies: category cleanup, domain-rule/placeholder -> NaN conversion,
de-duplication, mechanism-matched imputation, a leak-safe/deployable encoder/scaler
pipeline, and the train/test split itself, all in one place instead of split across
files -- there's exactly one obvious spot to look for "how does raw data become a
model-ready split."

See Practical/W3/notebooks/02_preprocessing.ipynb for the full walkthrough, including
the empirical grid that picked this week's `config.yaml`-recorded encoder/scaler pair.

Two things every function here respects, on purpose:
  - leak-safe: `clean_dataset` and `split_features_target` are target- and
    split-independent, so they're safe to run on the whole dataset before splitting.
    `split_train_test` is the boundary line -- everything after it (imputation,
    encoding, scaling, inside `build_preprocessor`'s ColumnTransformer) is fit only on
    the training fold, never on data it's about to be evaluated against.
  - deployable from day one: every function up to (not including) the split is
    target-column-agnostic -- `y` comes back as `None` and nothing else breaks when
    called on label-free inference data, which never gets split at all.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from category_encoders import CountEncoder

def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies validity rules and converts impossible values to NaN.
    """
    report_rows = []

    for column, bounds in rules.items():
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")

        lower_ok = (
            numeric >= bounds["min"]
            if "min" in bounds
            else pd.Series(True, index=numeric.index)
        )

        upper_ok = (
            numeric <= bounds["max"]
            if "max" in bounds
            else pd.Series(True, index=numeric.index)
        )

        violations = numeric.notna() & ~(lower_ok & upper_ok)

        report_rows.append({
            "column": column,
            "rule": bounds,
            "violations": int(violations.sum()),
        })

        df.loc[violations, column] = np.nan

    return pd.DataFrame(report_rows)

def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies diagnosis: category cleanup, domain-rule/placeholder -> NaN
    conversion, de-duplication, and redundant-column removal. Target-agnostic -- safe
    to call on label-free inference data, since none of this depends on a target column.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    # numeric columns that load as text purely because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=columns_to_drop)

    return out

def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    Training data only.
    Removes exact duplicate rows and repeated IDs.
    Must run before split_dev_test().
    """
    out = df.drop_duplicates()

    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")

    return out

def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each MNAR-diagnosed column, before that
    column gets imputed -- so a model can still see the pattern even though the fill
    value itself (median/mode) can't carry it. Target-agnostic."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Returns (X, y, extras). `y` is `None` and `extras` has no target column when called
    on label-free inference data -- nothing downstream requires the target to be present.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    return X, y, extras


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}

_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    ),

    "ordinal": lambda seed: OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ),

    "count": lambda seed: CountEncoder(
        handle_unknown=0,
        handle_missing=0,
    ),

    "target": lambda seed: TargetEncoder(
        target_type="binary",
        cv=StratifiedKFold(
            5,
            shuffle=True,
            random_state=seed,
        ),
    ),
}

def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Factory: builds a leak-safe ColumnTransformer for this week's chosen encoder/scaler
    pair -- read from `config.yaml`'s `preprocessing` section (found by the empirical
    grid in 02_preprocessing.ipynb, not hardcoded here). Every encoder tolerates unseen
    categories at transform time; fit happens on the training fold only, via the
    surrounding sklearn Pipeline's own fit/transform discipline.
    """
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[scaler_name]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    
    encoder = _ENCODERS[encoder_name](
    preprocessing_config.get("random_state")
    )

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])

    indicator_cols = [f"{c}_was_missing" for c in mnar_indicator_sources]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Split into development data and a locked final test set.

    The development set is used for cross-validation.
    The test set stays untouched until final testing.
    """

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X,
        y,
        extras,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return (
        X_dev,
        X_test,
        y_dev,
        y_test,
        extras_dev,
        extras_test,
    )