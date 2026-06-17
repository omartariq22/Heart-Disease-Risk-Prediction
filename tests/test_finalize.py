"""Tests for final model packaging and model-card generation."""

from pathlib import Path

import pandas as pd

from src.finalize import load_final_decision, run_finalization


def _write_final_decision_inputs(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": "Support Vector Machine",
                "selection_reason": "Highest recall, then ROC-AUC and F1 among tuned candidates",
                "best_params": '{"clf__C": 0.1, "clf__gamma": "scale", "clf__kernel": "linear"}',
                "best_refit_score": 0.921031,
                "accuracy_mean": 0.859014,
                "accuracy_std": 0.054669,
                "precision_mean": 0.846551,
                "precision_std": 0.058653,
                "recall_mean": 0.908262,
                "recall_std": 0.039463,
                "f1_mean": 0.875794,
                "f1_std": 0.046373,
                "roc_auc_mean": 0.921031,
                "roc_auc_std": 0.037555,
                "average_precision_mean": 0.928812,
                "average_precision_std": 0.034115,
            }
        ]
    ).to_csv(results_dir / "tuned_selected_model.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "Support Vector Machine",
                "operating_point": "default_0.50",
                "threshold": 0.50,
            },
            {
                "model": "Support Vector Machine",
                "operating_point": "recall_at_least_0.90",
                "threshold": 0.40,
            },
        ]
    ).to_csv(results_dir / "evaluation_operating_points.csv", index=False)


def test_load_final_decision_prefers_locked_recall_threshold(tmp_path) -> None:
    results_dir = tmp_path / "results"
    _write_final_decision_inputs(results_dir)

    decision = load_final_decision(results_dir)

    assert decision.model_name == "Support Vector Machine"
    assert decision.best_params["clf__kernel"] == "linear"
    assert decision.operating_point == "recall_at_least_0.90"
    assert decision.threshold == 0.40


def test_run_finalization_writes_model_card_metrics_and_local_model(tmp_path) -> None:
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    _write_final_decision_inputs(results_dir)

    result = run_finalization(
        results_dir=results_dir,
        reports_dir=reports_dir,
        models_dir=models_dir,
    )

    assert result.model_path.exists()
    assert result.model_path.name == "final_model.joblib"
    assert result.final_metrics.shape[0] == 2
    assert result.final_predictions.shape[0] == 61
    assert result.final_metrics["recall"].between(0, 1).all()
    assert (results_dir / "final_test_metrics.csv").exists()
    assert (results_dir / "final_test_predictions.csv").exists()
    assert (results_dir / "final_model_metadata.csv").exists()
    assert "Model Card" in result.model_card_path.read_text(encoding="utf-8")
    assert "Final Test Evaluation Report" in result.final_test_report_path.read_text(
        encoding="utf-8"
    )
