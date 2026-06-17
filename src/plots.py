"""Visualization QA, manifest, and summary plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.calibration import calibration_curve

from src.schema import RESULTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALIZATION_REPORT_PATH = REPORTS_DIR / "VISUALIZATION_REPORT.md"

REQUIRED_VISUALIZATIONS = [
    {
        "requirement": "Target class distribution bar chart",
        "figure": "eda_target_distribution.png",
        "status": "complete",
    },
    {
        "requirement": "Age distribution histogram by target",
        "figure": "eda_age_distribution_by_target.png",
        "status": "complete",
    },
    {
        "requirement": "Heart disease percentage by age group",
        "figure": "eda_age_group_target_rate.png",
        "status": "complete",
    },
    {
        "requirement": "Heart disease percentage by sex",
        "figure": "eda_sex_target_rate.png",
        "status": "complete",
    },
    {
        "requirement": "Pearson correlation heatmap",
        "figure": "eda_pearson_correlation_heatmap.png",
        "status": "complete",
    },
    {
        "requirement": "Box plots for outlier detection",
        "figure": "outliers_numeric_boxplots.png",
        "status": "complete",
    },
    {
        "requirement": "Chest pain type vs target count plot",
        "figure": "eda_chest_pain_vs_target.png",
        "status": "complete",
    },
    {
        "requirement": "Exercise-induced angina vs target count plot",
        "figure": "eda_exang_vs_target.png",
        "status": "complete",
    },
    {
        "requirement": "Maximum heart rate vs target box plot",
        "figure": "eda_thalach_by_target.png",
        "status": "complete",
    },
    {
        "requirement": "Cholesterol vs target box plot",
        "figure": "eda_cholesterol_by_target.png",
        "status": "complete",
    },
    {
        "requirement": "Mutual information feature relevance chart",
        "figure": "eda_mutual_information.png",
        "status": "complete",
    },
    {
        "requirement": "Model CV-score comparison with error bars",
        "figure": "visualization_cv_score_comparison_error_bars.png",
        "status": "complete",
    },
    {
        "requirement": "ROC curve overlay",
        "figure": "evaluation_roc_curves.png",
        "status": "complete",
    },
    {
        "requirement": "Precision-recall curve overlay",
        "figure": "evaluation_precision_recall_curves.png",
        "status": "complete",
    },
    {
        "requirement": "Confusion matrix heatmaps for top models",
        "figure": "evaluation_confusion_support_vector_machine_recall_at_least_0.90.png",
        "status": "complete",
    },
    {
        "requirement": "Feature importance chart for final candidate",
        "figure": "interpret_permutation_importance.png",
        "status": "complete",
    },
    {
        "requirement": "Calibration plot for final candidate",
        "figure": "visualization_svm_calibration_curve.png",
        "status": "complete_optional",
    },
]


@dataclass(frozen=True)
class VisualizationResult:
    """Structured visualization QA artifacts."""

    manifest: pd.DataFrame
    requirements: pd.DataFrame
    figure_paths: list[Path]


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def generate_cv_score_comparison_plot() -> Path:
    """Generate model CV comparison with +/- 1 std error bars."""
    summary = pd.read_csv(RESULTS_DIR / "baseline_model_comparison.csv")
    metrics = ["recall", "roc_auc", "f1"]
    plot_rows = []
    for _, row in summary.iterrows():
        for metric in metrics:
            plot_rows.append(
                {
                    "model": row["model"],
                    "metric": metric,
                    "mean": row[f"{metric}_mean"],
                    "std": row[f"{metric}_std"],
                }
            )
    plot_data = pd.DataFrame(plot_rows)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 5))
    axis = sns.barplot(data=plot_data, x="model", y="mean", hue="metric")
    model_order = plot_data["model"].drop_duplicates().tolist()
    metric_order = metrics
    patch_lookup = {
        (model, metric): patch
        for (model, metric), patch in zip(
            [(model, metric) for model in model_order for metric in metric_order],
            axis.patches,
        )
    }
    for _, row in plot_data.iterrows():
        patch = patch_lookup[(row["model"], row["metric"])]
        x = patch.get_x() + patch.get_width() / 2
        axis.errorbar(x, row["mean"], yerr=row["std"], color="black", capsize=3, linewidth=1)

    plt.title("Baseline 5-Fold CV Score Comparison With +/- 1 Std")
    plt.xlabel("Model")
    plt.ylabel("Cross-Validated Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    return _save_current_figure(FIGURES_DIR / "visualization_cv_score_comparison_error_bars.png")


def generate_calibration_plot() -> Path:
    """Generate calibration plot for the selected tuned SVM from OOF predictions."""
    predictions = pd.read_csv(RESULTS_DIR / "evaluation_oof_predictions.csv")
    svm = predictions.loc[predictions["model"].eq("Support Vector Machine")]
    fraction_positive, mean_predicted = calibration_curve(
        svm["y_true"],
        svm["y_score"],
        n_bins=8,
        strategy="uniform",
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_predicted, fraction_positive, marker="o", label="Tuned SVM")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.title("Calibration Curve - Tuned SVM OOF Predictions")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction Of Positives")
    plt.legend()
    return _save_current_figure(FIGURES_DIR / "visualization_svm_calibration_curve.png")


def build_figure_manifest() -> pd.DataFrame:
    """Build a manifest of generated figure files with dimensions."""
    rows = []
    for path in sorted(FIGURES_DIR.glob("*.png")):
        with Image.open(path) as image:
            width, height = image.size
        rows.append(
            {
                "figure": path.name,
                "relative_path": f"outputs/figures/{path.name}",
                "width_px": width,
                "height_px": height,
                "file_size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def build_requirements_table() -> pd.DataFrame:
    """Map required visualization deliverables to generated files."""
    requirements = pd.DataFrame(REQUIRED_VISUALIZATIONS)
    requirements["relative_path"] = "outputs/figures/" + requirements["figure"]
    requirements["exists"] = requirements["figure"].apply(lambda name: (FIGURES_DIR / name).exists())
    return requirements


def render_visualization_report(result: VisualizationResult) -> str:
    """Render visualization QA report."""
    complete_count = int(result.requirements["exists"].sum())
    total_count = len(result.requirements)

    return f"""# Visualization Report

## Scope

This report indexes all generated project figures, confirms required visualization coverage, and records image dimensions for documentation QA. All generated figures are PNG files intended for report insertion.

## Required Visualization Coverage

Covered required/optional visualizations: {complete_count}/{total_count}

{result.requirements.to_markdown(index=False)}

## Figure Manifest

{result.manifest.to_markdown(index=False)}

## Notes

- Core EDA plots, outlier diagnostics, model comparison figures, ROC/PR curves, confusion matrices, and feature-importance plots are available under `outputs/figures/`.
- `visualization_cv_score_comparison_error_bars.png` provides the explicit +/- 1 standard deviation CV comparison required for model reporting.
- `visualization_svm_calibration_curve.png` is optional but useful for discussing probability calibration before final threshold selection.
"""


def write_visualization_outputs(result: VisualizationResult) -> None:
    """Persist visualization manifest and report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.manifest.to_csv(RESULTS_DIR / "visualization_figure_manifest.csv", index=False)
    result.requirements.to_csv(
        RESULTS_DIR / "visualization_requirements_coverage.csv", index=False
    )
    VISUALIZATION_REPORT_PATH.write_text(render_visualization_report(result), encoding="utf-8")


def run_visualization_qa() -> VisualizationResult:
    """Generate Step 12 visualization QA artifacts."""
    generated = [
        generate_cv_score_comparison_plot(),
        generate_calibration_plot(),
    ]
    manifest = build_figure_manifest()
    requirements = build_requirements_table()
    result = VisualizationResult(
        manifest=manifest,
        requirements=requirements,
        figure_paths=generated,
    )
    write_visualization_outputs(result)
    return result


if __name__ == "__main__":
    run_visualization_qa()
