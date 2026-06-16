"""Tests for training-only outlier detection."""

from src.data import load_raw_data
from src.outliers import (
    calculate_iqr_outliers,
    calculate_zscore_outliers,
)
from src.preprocess import clean_heart_data
from src.split import stratified_train_test_split


def test_iqr_outlier_counts_match_training_split() -> None:
    cleaned = clean_heart_data(load_raw_data()).cleaned
    split = stratified_train_test_split(cleaned)

    summary, records = calculate_iqr_outliers(split.train)

    counts = summary.set_index("feature")["outlier_count"].to_dict()
    assert counts == {
        "age": 0,
        "trestbps": 9,
        "chol": 3,
        "thalach": 0,
        "oldpeak": 2,
    }
    assert len(records) == 14


def test_zscore_outlier_counts_match_training_split() -> None:
    cleaned = clean_heart_data(load_raw_data()).cleaned
    split = stratified_train_test_split(cleaned)

    summary, records = calculate_zscore_outliers(split.train)

    counts = summary.set_index("feature")["outlier_count"].to_dict()
    assert counts == {
        "age": 0,
        "trestbps": 2,
        "chol": 3,
        "thalach": 0,
        "oldpeak": 2,
    }
    assert len(records) == 7
