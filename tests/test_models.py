"""Tests for baseline model inventory and CV outputs."""

from src.models import build_model_pipeline, build_model_specs, evaluate_baseline_models


def test_model_inventory_contains_required_classifiers() -> None:
    specs = build_model_specs()
    names = [spec.name for spec in specs]

    assert names == [
        "Dummy Stratified",
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Support Vector Machine",
        "Decision Tree",
        "Random Forest",
        "Gaussian Naive Bayes",
        "Gradient Boosting",
        "XGBoost",
    ]


def test_model_pipeline_uses_shared_preprocessor_and_classifier_step() -> None:
    spec = build_model_specs()[1]

    pipeline = build_model_pipeline(spec)

    assert list(pipeline.named_steps.keys()) == ["prep", "clf"]
    assert pipeline.named_steps["clf"].__class__.__name__ == "LogisticRegression"


def test_baseline_cv_outputs_expected_rows_and_metrics() -> None:
    result = evaluate_baseline_models()

    assert result.fold_results.shape[0] == 45
    assert result.summary.shape[0] == 9
    assert "XGBoost" in result.summary["model"].tolist()
    assert result.summary["recall_mean"].between(0, 1).all()
    assert result.summary["roc_auc_mean"].between(0, 1).all()
    assert result.summary["average_precision_mean"].between(0, 1).all()
