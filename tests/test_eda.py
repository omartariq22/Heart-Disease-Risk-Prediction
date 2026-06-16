"""Tests for train-only EDA helpers."""

from src.data import load_raw_data
from src.eda import add_age_groups, target_rate_table
from src.preprocess import clean_heart_data
from src.split import stratified_train_test_split


def test_stratified_split_preserves_expected_sizes_and_target_balance() -> None:
    cleaned = clean_heart_data(load_raw_data()).cleaned

    split = stratified_train_test_split(cleaned)

    assert split.train.shape == (241, 14)
    assert split.test.shape == (61, 14)
    assert split.train["target"].value_counts().sort_index().to_dict() == {0: 110, 1: 131}
    assert split.test["target"].value_counts().sort_index().to_dict() == {0: 28, 1: 33}


def test_age_group_rates_are_computed_from_training_split_only() -> None:
    cleaned = clean_heart_data(load_raw_data()).cleaned
    split = stratified_train_test_split(cleaned)
    train = add_age_groups(split.train)

    age_rates = target_rate_table(train, "age_group")
    younger = age_rates.loc[age_rates["age_group"].eq("<40")].iloc[0]

    assert int(age_rates["rows"].sum()) == len(split.train)
    assert int(younger["rows"]) == 12
    assert int(younger["heart_disease_cases"]) == 9
    assert float(younger["heart_disease_percentage"]) == 75.0
