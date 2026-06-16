"""Tests for deterministic data-cleaning behavior."""

import numpy as np

from src.data import load_raw_data
from src.preprocess import clean_heart_data


def test_clean_heart_data_removes_duplicate_and_decodes_sentinels() -> None:
    raw = load_raw_data()

    result = clean_heart_data(raw)

    assert result.cleaned.shape == (302, 14)
    assert result.cleaned.duplicated().sum() == 0
    assert result.cleaned["ca"].isna().sum() == 4
    assert result.cleaned["thal"].isna().sum() == 2
    assert not result.cleaned["ca"].eq(4).any()
    assert not result.cleaned["thal"].eq(0).any()


def test_clean_heart_data_preserves_schema_and_target_counts() -> None:
    raw = load_raw_data()

    result = clean_heart_data(raw)

    assert list(result.cleaned.columns) == list(raw.columns)
    assert result.cleaned["target"].value_counts().sort_index().to_dict() == {0: 138, 1: 164}
    assert np.isclose(result.cleaned["oldpeak"].max(), 6.2)


def test_clean_heart_data_fails_on_unexpected_schema() -> None:
    raw = load_raw_data()
    malformed = raw.drop(columns=["target"])

    try:
        clean_heart_data(malformed)
    except ValueError as exc:
        assert "Unexpected dataset schema" in str(exc)
    else:
        raise AssertionError("Expected schema validation to fail.")
