"""Tests for hyperparameter tuning helpers."""

from src.tune import build_search_space_table, get_tuning_candidates, tune_candidate_models


def test_tuning_candidate_shortlist_is_professional_top_three() -> None:
    candidates = get_tuning_candidates()
    names = [candidate.name for candidate in candidates]

    assert names == ["Logistic Regression", "Support Vector Machine", "XGBoost"]


def test_search_space_combination_counts_are_expected() -> None:
    search_space = build_search_space_table().set_index("model")

    assert int(search_space.loc["Logistic Regression", "n_parameter_combinations"]) == 4
    assert int(search_space.loc["Support Vector Machine", "n_parameter_combinations"]) == 12
    assert int(search_space.loc["XGBoost", "n_parameter_combinations"]) == 8


def test_single_model_tuning_smoke_test() -> None:
    result = tune_candidate_models(candidate_names=["Logistic Regression"])

    assert result.tuned_comparison.shape[0] == 1
    assert result.selected_model["model"].iloc[0] == "Logistic Regression"
    assert result.full_cv_results["model"].eq("Logistic Regression").all()
    assert result.tuned_comparison["roc_auc_mean"].between(0, 1).all()
