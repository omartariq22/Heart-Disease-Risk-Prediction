"""Tests for visualization QA and reporting helpers."""

from pathlib import Path

import pandas as pd
from PIL import Image

from src import plots


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 16), color="white").save(path)


def test_visualization_qa_writes_complete_manifest_and_report(tmp_path, monkeypatch) -> None:
    figures_dir = tmp_path / "figures"
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"

    for requirement in plots.REQUIRED_VISUALIZATIONS:
        if not requirement["figure"].startswith("visualization_"):
            _write_png(figures_dir / requirement["figure"])

    results_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["Logistic Regression", "Support Vector Machine"],
            "recall_mean": [0.87, 0.89],
            "recall_std": [0.05, 0.04],
            "roc_auc_mean": [0.91, 0.90],
            "roc_auc_std": [0.03, 0.03],
            "f1_mean": [0.86, 0.85],
            "f1_std": [0.04, 0.04],
        }
    ).to_csv(results_dir / "baseline_model_comparison.csv", index=False)
    pd.DataFrame(
        {
            "model": ["Support Vector Machine"] * 8,
            "y_true": [0, 0, 0, 1, 1, 1, 0, 1],
            "y_score": [0.05, 0.15, 0.30, 0.45, 0.70, 0.88, 0.62, 0.96],
        }
    ).to_csv(results_dir / "evaluation_oof_predictions.csv", index=False)

    monkeypatch.setattr(plots, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(plots, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(plots, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(
        plots,
        "VISUALIZATION_REPORT_PATH",
        reports_dir / "VISUALIZATION_REPORT.md",
    )

    result = plots.run_visualization_qa()

    assert len(plots.REQUIRED_VISUALIZATIONS) == 17
    assert result.requirements["exists"].all()
    assert result.manifest["width_px"].gt(0).all()
    assert result.manifest["height_px"].gt(0).all()
    assert (figures_dir / "visualization_cv_score_comparison_error_bars.png").exists()
    assert (figures_dir / "visualization_svm_calibration_curve.png").exists()
    assert (results_dir / "visualization_figure_manifest.csv").exists()
    assert (results_dir / "visualization_requirements_coverage.csv").exists()
    assert "Covered required/optional visualizations: 17/17" in (
        reports_dir / "VISUALIZATION_REPORT.md"
    ).read_text(encoding="utf-8")
