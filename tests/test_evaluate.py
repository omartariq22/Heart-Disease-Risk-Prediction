"""Tests for model evaluation helpers."""

import numpy as np
import pandas as pd

from src.evaluate import (
    build_oof_metrics,
    build_oof_predictions,
    build_threshold_sweep,
    calculate_metrics,
    select_operating_points,
)


def test_calculate_metrics_uses_threshold_and_confusion_counts() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_score = np.array([0.1, 0.7, 0.8, 0.2])

    metrics = calculate_metrics(y_true, y_score, threshold=0.5)

    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5


def test_oof_prediction_generation_for_single_model_has_training_rows() -> None:
    predictions, _, split_summary = build_oof_predictions(selected_model_names=["Dummy Stratified"])

    assert predictions.shape[0] == 241
    assert predictions["model"].unique().tolist() == ["Dummy Stratified"]
    assert predictions["y_score"].between(0, 1).all()
    assert split_summary.loc[split_summary["split"].eq("train"), "rows"].iloc[0] == 241


def test_threshold_operating_points_include_target_recall_when_available() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["Toy"] * 6,
            "row_index_in_training_split": range(6),
            "y_true": [0, 0, 0, 1, 1, 1],
            "y_score": [0.1, 0.2, 0.8, 0.4, 0.7, 0.9],
        }
    )

    metrics = build_oof_metrics(predictions)
    sweep = build_threshold_sweep(predictions, ["Toy"])
    operating_points = select_operating_points(metrics, sweep, target_recall=0.90)

    assert "default_0.50" in operating_points["operating_point"].tolist()
    assert "max_f1" in operating_points["operating_point"].tolist()
    assert "recall_at_least_0.90" in operating_points["operating_point"].tolist()
