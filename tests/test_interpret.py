"""Tests for tuned-model interpretation helpers."""

import pytest

from src.data import load_raw_data
from src.interpret import (
    build_tuned_pipeline,
    extract_linear_coefficients,
    extract_xgboost_importances,
)
from src.model_preprocess import build_feature_target_split
from src.preprocess import clean_heart_data


def test_tuned_linear_svm_coefficients_have_expected_feature_count() -> None:
    cleaned = clean_heart_data(load_raw_data()).cleaned
    split = build_feature_target_split(cleaned)
    pipeline = build_tuned_pipeline("Support Vector Machine")

    pipeline.fit(split.x_train, split.y_train)
    coefficients = extract_linear_coefficients(pipeline, "Support Vector Machine")

    assert coefficients.shape[0] == 22
    assert coefficients["abs_coefficient"].is_monotonic_decreasing
    assert {"feature", "coefficient", "direction"}.issubset(coefficients.columns)


def test_tuned_xgboost_importances_have_expected_feature_count() -> None:
    cleaned = clean_heart_data(load_raw_data()).cleaned
    split = build_feature_target_split(cleaned)
    pipeline = build_tuned_pipeline("XGBoost")

    pipeline.fit(split.x_train, split.y_train)
    importances = extract_xgboost_importances(pipeline)

    assert importances.shape[0] == 22
    assert importances["importance"].ge(0).all()
    assert importances["importance"].sum() > 0


def test_build_tuned_pipeline_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported tuned model"):
        build_tuned_pipeline("Unknown Model")
