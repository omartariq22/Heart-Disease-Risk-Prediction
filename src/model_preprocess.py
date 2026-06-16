"""Leak-proof modelling preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_raw_data
from src.preprocess import clean_heart_data
from src.schema import (
    BINARY_COLUMNS,
    NOMINAL_CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    ORDINAL_COUNT_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMN,
)
from src.split import DataSplit, stratified_train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
PREPROCESSING_REPORT_PATH = REPORTS_DIR / "PREPROCESSING_PIPELINE_REPORT.md"

BINARY_FEATURES = [column for column in BINARY_COLUMNS if column != TARGET_COLUMN]


@dataclass(frozen=True)
class FeatureTargetSplit:
    """Container for model-ready feature/target splits."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    split: DataSplit


@dataclass(frozen=True)
class PreprocessingArtifacts:
    """Inspection artifacts for the shared preprocessing pipeline."""

    feature_target_split: FeatureTargetSplit
    preprocessor: ColumnTransformer
    column_routing: pd.DataFrame
    transformed_feature_names: list[str]
    transformed_train_shape: tuple[int, int]
    transformed_test_shape: tuple[int, int]
    missing_values_before_preprocessing: pd.DataFrame


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the binary target."""
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def build_feature_target_split(df: pd.DataFrame) -> FeatureTargetSplit:
    """Create the project train/test split and separate X/y."""
    split = stratified_train_test_split(df)
    x_train, y_train = split_features_target(split.train)
    x_test, y_test = split_features_target(split.test)
    return FeatureTargetSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        split=split,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build the shared preprocessing transformer used by every classifier."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
            ),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERICAL_COLUMNS),
            ("nominal", nominal_pipeline, NOMINAL_CATEGORICAL_COLUMNS),
            ("ordinal", ordinal_pipeline, ORDINAL_COUNT_COLUMNS),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_column_routing_table() -> pd.DataFrame:
    """Document how each input feature is handled by the ColumnTransformer."""
    rows = []
    for column in NUMERICAL_COLUMNS:
        rows.append(
            {
                "feature": column,
                "group": "numeric",
                "pipeline_steps": "SimpleImputer(strategy='median') -> StandardScaler()",
            }
        )
    for column in NOMINAL_CATEGORICAL_COLUMNS:
        rows.append(
            {
                "feature": column,
                "group": "nominal",
                "pipeline_steps": (
                    "SimpleImputer(strategy='most_frequent') -> "
                    "OneHotEncoder(handle_unknown='ignore', drop='if_binary')"
                ),
            }
        )
    for column in ORDINAL_COUNT_COLUMNS:
        rows.append(
            {
                "feature": column,
                "group": "ordinal",
                "pipeline_steps": "SimpleImputer(strategy='most_frequent')",
            }
        )
    for column in BINARY_FEATURES:
        rows.append(
            {
                "feature": column,
                "group": "binary",
                "pipeline_steps": "passthrough",
            }
        )
    return pd.DataFrame(rows)


def missing_values_before_preprocessing(x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values that will be handled inside the pipeline."""
    return pd.DataFrame(
        {
            "feature": x_train.columns,
            "train_missing_count": x_train.isna().sum().values,
            "test_missing_count": x_test.isna().sum().values,
        }
    )


def inspect_preprocessor(feature_target_split: FeatureTargetSplit) -> PreprocessingArtifacts:
    """Fit the preprocessor on train only and collect inspection artifacts."""
    preprocessor = build_preprocessor()
    x_train_transformed = preprocessor.fit_transform(feature_target_split.x_train)
    x_test_transformed = preprocessor.transform(feature_target_split.x_test)
    transformed_feature_names = preprocessor.get_feature_names_out().tolist()

    return PreprocessingArtifacts(
        feature_target_split=feature_target_split,
        preprocessor=preprocessor,
        column_routing=build_column_routing_table(),
        transformed_feature_names=transformed_feature_names,
        transformed_train_shape=x_train_transformed.shape,
        transformed_test_shape=x_test_transformed.shape,
        missing_values_before_preprocessing=missing_values_before_preprocessing(
            feature_target_split.x_train,
            feature_target_split.x_test,
        ),
    )


def render_preprocessing_report(artifacts: PreprocessingArtifacts) -> str:
    """Render the shared preprocessing pipeline report."""
    feature_names = pd.DataFrame({"transformed_feature": artifacts.transformed_feature_names})
    missing_values = artifacts.missing_values_before_preprocessing

    return f"""# Preprocessing Pipeline Report

## Scope

This report documents the shared preprocessing `ColumnTransformer` that every classification model will use. The transformer is fitted on the training split only and then applied to validation/test data through scikit-learn `Pipeline` objects.

## Split Summary

{artifacts.feature_target_split.split.summary.to_markdown(index=False)}

## Feature And Target Shapes

| object | shape |
|:--|:--|
| `X_train` | {artifacts.feature_target_split.x_train.shape} |
| `X_test` | {artifacts.feature_target_split.x_test.shape} |
| `y_train` | {artifacts.feature_target_split.y_train.shape} |
| `y_test` | {artifacts.feature_target_split.y_test.shape} |
| transformed training matrix | {artifacts.transformed_train_shape} |
| transformed test matrix | {artifacts.transformed_test_shape} |

## Column Routing

{artifacts.column_routing.to_markdown(index=False)}

## Missing Values Before Pipeline Imputation

{missing_values.to_markdown(index=False)}

## Transformed Feature Names

{feature_names.to_markdown(index=False)}

## Leakage-Control Notes

- Deterministic cleaning happens before splitting because duplicate removal and sentinel decoding do not estimate parameters from the data.
- Median, most-frequent imputation, scaling, and one-hot encoding are fitted inside the `ColumnTransformer`.
- Later model training must wrap this preprocessor inside `Pipeline([('prep', preprocessor), ('clf', estimator)])`.
- The held-out test set is transformed only by a preprocessor fitted on the training data.
- Age groups remain EDA-only and are not included as model features.
"""


def write_preprocessing_outputs(artifacts: PreprocessingArtifacts) -> None:
    """Persist preprocessing inspection tables and report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts.feature_target_split.split.summary.to_csv(
        RESULTS_DIR / "preprocess_split_summary.csv", index=False
    )
    artifacts.column_routing.to_csv(RESULTS_DIR / "preprocess_column_routing.csv", index=False)
    pd.DataFrame({"transformed_feature": artifacts.transformed_feature_names}).to_csv(
        RESULTS_DIR / "preprocess_transformed_feature_names.csv", index=False
    )
    artifacts.missing_values_before_preprocessing.to_csv(
        RESULTS_DIR / "preprocess_missing_values_before_pipeline.csv", index=False
    )
    PREPROCESSING_REPORT_PATH.write_text(render_preprocessing_report(artifacts), encoding="utf-8")


def build_preprocessing_artifacts() -> PreprocessingArtifacts:
    """Build and inspect the Step 7 preprocessing artifacts."""
    raw = load_raw_data()
    cleaned = clean_heart_data(raw).cleaned
    feature_target_split = build_feature_target_split(cleaned)
    return inspect_preprocessor(feature_target_split)


def run_preprocessing_inspection() -> PreprocessingArtifacts:
    """Run Step 7 preprocessing inspection and persist outputs."""
    artifacts = build_preprocessing_artifacts()
    write_preprocessing_outputs(artifacts)
    return artifacts


if __name__ == "__main__":
    run_preprocessing_inspection()
