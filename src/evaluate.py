"""Model evaluation utilities built on out-of-fold training predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import SEED
from src.data import load_raw_data
from src.model_preprocess import build_feature_target_split
from src.models import build_model_pipeline, build_model_specs, evaluate_baseline_models
from src.preprocess import clean_heart_data
from src.schema import RESULTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_EVALUATION_REPORT_PATH = REPORTS_DIR / "MODEL_EVALUATION_REPORT.md"

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
TARGET_RECALL = 0.90


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Structured Step 9 model evaluation outputs."""

    baseline_summary: pd.DataFrame
    split_summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    oof_metrics: pd.DataFrame
    threshold_sweep: pd.DataFrame
    operating_points: pd.DataFrame
    confusion_matrices: pd.DataFrame
    figure_paths: list[Path]


def calculate_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculate classification metrics for a probability threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6),
        "average_precision": round(float(average_precision_score(y_true, y_score)), 6),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def build_oof_predictions(selected_model_names: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate out-of-fold probabilities for baseline models on the training split."""
    raw = load_raw_data()
    cleaned = clean_heart_data(raw).cleaned
    feature_target_split = build_feature_target_split(cleaned)
    model_specs = build_model_specs()
    if selected_model_names is not None:
        model_specs = [spec for spec in model_specs if spec.name in selected_model_names]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for spec in model_specs:
        pipeline = build_model_pipeline(spec)
        probabilities = cross_val_predict(
            pipeline,
            feature_target_split.x_train,
            feature_target_split.y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        for index, (target, probability) in enumerate(
            zip(feature_target_split.y_train.to_numpy(), probabilities)
        ):
            rows.append(
                {
                    "model": spec.name,
                    "row_index_in_training_split": index,
                    "y_true": int(target),
                    "y_score": round(float(probability), 8),
                }
            )

    return (
        pd.DataFrame(rows),
        feature_target_split.y_train.reset_index(drop=True).to_frame("y_true"),
        feature_target_split.split.summary,
    )


def build_oof_metrics(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate default-threshold metrics for each model from OOF predictions."""
    rows = []
    for model, group in oof_predictions.groupby("model", sort=False):
        metrics = calculate_metrics(group["y_true"], group["y_score"].to_numpy(), threshold=0.5)
        rows.append({"model": model, **metrics})
    return (
        pd.DataFrame(rows)
        .sort_values(["recall", "roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )


def build_threshold_sweep(
    oof_predictions: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    """Sweep probability thresholds for selected models."""
    rows = []
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    selected_predictions = oof_predictions.loc[oof_predictions["model"].isin(model_names)]

    for model, group in selected_predictions.groupby("model", sort=False):
        y_true = group["y_true"]
        y_score = group["y_score"].to_numpy()
        for threshold in thresholds:
            metrics = calculate_metrics(y_true, y_score, threshold=float(threshold))
            rows.append({"model": model, **metrics})

    return pd.DataFrame(rows)


def select_operating_points(
    oof_metrics: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
    target_recall: float = TARGET_RECALL,
) -> pd.DataFrame:
    """Select default, max-F1, and target-recall operating points."""
    rows = []
    for _, row in oof_metrics.iterrows():
        rows.append({"model": row["model"], "operating_point": "default_0.50", **row.to_dict()})

    for model, group in threshold_sweep.groupby("model", sort=False):
        max_f1 = group.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
        rows.append({"model": model, "operating_point": "max_f1", **max_f1.to_dict()})

        recall_candidates = group.loc[group["recall"].ge(target_recall)]
        if not recall_candidates.empty:
            target_row = recall_candidates.sort_values(
                ["f1", "precision", "threshold"], ascending=[False, False, False]
            ).iloc[0]
            rows.append(
                {
                    "model": model,
                    "operating_point": f"recall_at_least_{target_recall:.2f}",
                    **target_row.to_dict(),
                }
            )

    operating_points = pd.DataFrame(rows)
    metric_columns = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]
    return operating_points[["model", "operating_point", *metric_columns]]


def build_confusion_matrices(operating_points: pd.DataFrame) -> pd.DataFrame:
    """Reshape selected operating-point confusion matrices for plotting/reporting."""
    rows = []
    for _, row in operating_points.iterrows():
        for actual, predicted, count in [
            (0, 0, row["true_negative"]),
            (0, 1, row["false_positive"]),
            (1, 0, row["false_negative"]),
            (1, 1, row["true_positive"]),
        ]:
            rows.append(
                {
                    "model": row["model"],
                    "operating_point": row["operating_point"],
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def generate_evaluation_figures(
    oof_predictions: pd.DataFrame,
    oof_metrics: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
    operating_points: pd.DataFrame,
) -> list[Path]:
    """Generate Step 9 model evaluation figures."""
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    metric_plot = oof_metrics.melt(
        id_vars="model",
        value_vars=["recall", "roc_auc", "average_precision", "f1", "precision", "accuracy"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(11, 5))
    sns.barplot(data=metric_plot, x="model", y="score", hue="metric")
    plt.title("Out-of-Fold Metric Comparison - Default Threshold")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=35, ha="right")
    paths.append(_save_current_figure(FIGURES_DIR / "evaluation_oof_metric_comparison.png"))

    plt.figure(figsize=(7, 5))
    for model, group in oof_predictions.groupby("model", sort=False):
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_score"])
        auc_value = roc_auc_score(group["y_true"], group["y_score"])
        plt.plot(fpr, tpr, label=f"{model} (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.title("ROC Curves From Out-of-Fold Predictions")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=7)
    paths.append(_save_current_figure(FIGURES_DIR / "evaluation_roc_curves.png"))

    plt.figure(figsize=(7, 5))
    for model, group in oof_predictions.groupby("model", sort=False):
        precision, recall, _ = precision_recall_curve(group["y_true"], group["y_score"])
        ap_value = average_precision_score(group["y_true"], group["y_score"])
        plt.plot(recall, precision, label=f"{model} (AP={ap_value:.3f})")
    plt.title("Precision-Recall Curves From Out-of-Fold Predictions")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=7)
    paths.append(_save_current_figure(FIGURES_DIR / "evaluation_precision_recall_curves.png"))

    if not threshold_sweep.empty:
        sweep_plot = threshold_sweep.melt(
            id_vars=["model", "threshold"],
            value_vars=["precision", "recall", "f1", "specificity"],
            var_name="metric",
            value_name="score",
        )
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=sweep_plot, x="threshold", y="score", hue="metric", style="model")
        plt.axhline(TARGET_RECALL, linestyle="--", color="gray", linewidth=1)
        plt.title("Threshold Sweep For Top Baseline Models")
        plt.xlabel("Probability Threshold")
        plt.ylabel("Score")
        plt.ylim(0, 1.02)
        paths.append(_save_current_figure(FIGURES_DIR / "evaluation_threshold_sweep.png"))

    threshold_models = threshold_sweep["model"].drop_duplicates().tolist()
    selected_confusions = operating_points.loc[
        operating_points["model"].isin(threshold_models)
        & operating_points["operating_point"].isin(["default_0.50", "recall_at_least_0.90"])
    ]
    for _, row in selected_confusions.iterrows():
        matrix = np.array(
            [
                [row["true_negative"], row["false_positive"]],
                [row["false_negative"], row["true_positive"]],
            ]
        )
        plt.figure(figsize=(4.5, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"],
        )
        plt.title(f"{row['model']} - {row['operating_point']}")
        paths.append(
            _save_current_figure(
                FIGURES_DIR
                / f"evaluation_confusion_{row['model'].lower().replace(' ', '_').replace('-', '')}_{row['operating_point']}.png"
            )
        )

    return paths


def build_model_evaluation() -> ModelEvaluationResult:
    """Build all Step 9 evaluation artifacts."""
    baseline = evaluate_baseline_models()
    top_models = baseline.summary["model"].head(2).tolist()
    oof_predictions, _, split_summary = build_oof_predictions()
    oof_metrics = build_oof_metrics(oof_predictions)
    threshold_sweep = build_threshold_sweep(oof_predictions, top_models)
    operating_points = select_operating_points(oof_metrics, threshold_sweep)
    confusion_matrices = build_confusion_matrices(operating_points)
    figure_paths = generate_evaluation_figures(
        oof_predictions,
        oof_metrics,
        threshold_sweep,
        operating_points,
    )

    return ModelEvaluationResult(
        baseline_summary=baseline.summary,
        split_summary=split_summary,
        oof_predictions=oof_predictions,
        oof_metrics=oof_metrics,
        threshold_sweep=threshold_sweep,
        operating_points=operating_points,
        confusion_matrices=confusion_matrices,
        figure_paths=figure_paths,
    )


def render_evaluation_report(result: ModelEvaluationResult) -> str:
    """Render Step 9 model evaluation report."""
    top_default = result.oof_metrics.iloc[0]
    tuned_points = result.operating_points.loc[
        result.operating_points["operating_point"].str.startswith("recall_at_least")
    ]
    figure_list = "\n".join(f"- `outputs/figures/{path.name}`" for path in result.figure_paths)

    return f"""# Model Evaluation Report

## Scope

This evaluation uses out-of-fold predictions from stratified 5-fold cross-validation on the training split only. The held-out test set is intentionally not evaluated in Step 9 because hyperparameter tuning has not been completed yet.

## Split Summary

{result.split_summary.to_markdown(index=False)}

## Baseline Cross-Validation Ranking

{result.baseline_summary.to_markdown(index=False)}

## Out-of-Fold Metrics At Default Threshold 0.50

{result.oof_metrics.to_markdown(index=False)}

## Threshold Sweep Operating Points

{result.operating_points.to_markdown(index=False)}

## Tuned Recall Operating Points

{tuned_points.to_markdown(index=False) if not tuned_points.empty else "No threshold met the target recall constraint."}

## Confusion Matrix Counts

{result.confusion_matrices.to_markdown(index=False)}

## Exported Figures

{figure_list}

## Interpretation

- Best default-threshold OOF recall: {top_default["model"]} with recall {top_default["recall"]:.3f}.
- False negatives are the most clinically concerning error because they represent heart disease-positive patients predicted as negative.
- Threshold tuning demonstrates the trade-off between sensitivity and false alarms before final model selection.
- Final held-out test evaluation is deferred until Step 10 tuning is locked, preserving the test set as an unbiased final check.
"""


def write_model_evaluation_outputs(result: ModelEvaluationResult) -> None:
    """Persist Step 9 evaluation tables, figures, and report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    result.oof_predictions.to_csv(RESULTS_DIR / "evaluation_oof_predictions.csv", index=False)
    result.oof_metrics.to_csv(RESULTS_DIR / "evaluation_oof_metrics.csv", index=False)
    result.threshold_sweep.to_csv(RESULTS_DIR / "evaluation_threshold_sweep.csv", index=False)
    result.operating_points.to_csv(RESULTS_DIR / "evaluation_operating_points.csv", index=False)
    result.confusion_matrices.to_csv(RESULTS_DIR / "evaluation_confusion_matrices.csv", index=False)
    MODEL_EVALUATION_REPORT_PATH.write_text(render_evaluation_report(result), encoding="utf-8")


def run_model_evaluation() -> ModelEvaluationResult:
    """Run Step 9 model evaluation and persist outputs."""
    result = build_model_evaluation()
    write_model_evaluation_outputs(result)
    return result


if __name__ == "__main__":
    run_model_evaluation()
