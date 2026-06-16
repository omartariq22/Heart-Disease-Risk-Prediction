"""Tests for the shared modelling preprocessor."""

import numpy as np

from src.data import load_raw_data
from src.model_preprocess import build_feature_target_split, inspect_preprocessor
from src.preprocess import clean_heart_data


def test_preprocessor_routes_columns_and_transforms_expected_shape() -> None:
    cleaned = clean_heart_data(load_raw_data()).cleaned
    split = build_feature_target_split(cleaned)

    artifacts = inspect_preprocessor(split)

    assert artifacts.feature_target_split.x_train.shape == (241, 13)
    assert artifacts.feature_target_split.x_test.shape == (61, 13)
    assert artifacts.transformed_train_shape == (241, 22)
    assert artifacts.transformed_test_shape == (61, 22)
    assert len(artifacts.transformed_feature_names) == 22


def test_preprocessor_imputes_missing_values_inside_transformer() -> None:
    cleaned = clean_heart_data(load_raw_data()).cleaned
    split = build_feature_target_split(cleaned)

    artifacts = inspect_preprocessor(split)
    transformed = artifacts.preprocessor.transform(split.x_train)

    assert not np.isnan(transformed).any()
    missing_counts = artifacts.missing_values_before_preprocessing.set_index("feature")
    assert int(missing_counts.loc["ca", "train_missing_count"]) == 4
    assert int(missing_counts.loc["thal", "train_missing_count"]) == 1


def test_preprocessor_column_routing_matches_project_contract() -> None:
    cleaned = clean_heart_data(load_raw_data()).cleaned
    split = build_feature_target_split(cleaned)

    artifacts = inspect_preprocessor(split)
    routing = artifacts.column_routing.set_index("feature")["group"].to_dict()

    assert routing["age"] == "numeric"
    assert routing["oldpeak"] == "numeric"
    assert routing["cp"] == "nominal"
    assert routing["thal"] == "nominal"
    assert routing["ca"] == "ordinal"
    assert routing["sex"] == "binary"
    assert routing["fbs"] == "binary"
    assert routing["exang"] == "binary"
