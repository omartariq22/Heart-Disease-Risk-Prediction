"""Final model fitting, held-out test evaluation, and model-card generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import SEED
from src.data import load_raw_data
from src.evaluate import TARGET_RECALL, calculate_metrics
from src.model_preprocess import FeatureTargetSplit, build_feature_target_split
from src.models import build_model_pipeline, build_model_specs
from src.preprocess import clean_heart_data
from src.schema import RESULTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
MODEL_CARD_PATH = REPORTS_DIR / "MODEL_CARD.md"
FINAL_TEST_REPORT_PATH = REPORTS_DIR / "FINAL_TEST_EVALUATION_REPORT.md"


@dataclass(frozen=True)
class FinalModelDecision:
    """Locked final model and threshold decision."""

    model_name: str
    best_params: dict
    operating_point: str
    threshold: float
    cv_recall: float
    cv_roc_auc: float
    cv_f1: float


@dataclass(frozen=True)
class FinalizationResult:
    """Final model evaluation and persistence artifacts."""

    decision: FinalModelDecision
    split_summary: pd.DataFrame
    final_metrics: pd.DataFrame
    final_predictions: pd.DataFrame
    model_metadata: pd.DataFrame
    model_path: Path
    model_card_path: Path
    final_test_report_path: Path


def _display_path(path: Path) -> str:
    """Return a Markdown-friendly path for project or temporary artifacts."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_final_decision(results_dir: Path = RESULTS_DIR) -> FinalModelDecision:
    """Load the selected tuned model and locked threshold from prior steps."""
    selected = pd.read_csv(results_dir / "tuned_selected_model.csv").iloc[0]
    operating_points = pd.read_csv(results_dir / "evaluation_operating_points.csv")
    model_points = operating_points.loc[operating_points["model"].eq(selected["model"])]

    preferred_name = f"recall_at_least_{TARGET_RECALL:.2f}"
    for candidate in [preferred_name, "max_f1", "default_0.50"]:
        match = model_points.loc[model_points["operating_point"].eq(candidate)]
        if not match.empty:
            operating_point = match.iloc[0]
            break
    else:
        raise ValueError(f"No operating point found for selected model: {selected['model']}")

    return FinalModelDecision(
        model_name=str(selected["model"]),
        best_params=json.loads(selected["best_params"]),
        operating_point=str(operating_point["operating_point"]),
        threshold=float(operating_point["threshold"]),
        cv_recall=float(selected["recall_mean"]),
        cv_roc_auc=float(selected["roc_auc_mean"]),
        cv_f1=float(selected["f1_mean"]),
    )


def build_final_pipeline(decision: FinalModelDecision):
    """Build the selected tuned pipeline with locked hyperparameters."""
    specs = {spec.name: spec for spec in build_model_specs()}
    if decision.model_name not in specs:
        raise ValueError(f"Unsupported final model: {decision.model_name}")

    pipeline = build_model_pipeline(specs[decision.model_name])
    pipeline.set_params(**decision.best_params)
    return pipeline


def build_final_test_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    decision: FinalModelDecision,
) -> pd.DataFrame:
    """Evaluate default and locked final thresholds on the held-out test set."""
    rows = []
    default_metrics = calculate_metrics(y_true, y_score, threshold=0.50)
    rows.append(
        {
            "model": decision.model_name,
            "evaluation_split": "held_out_test",
            "operating_point": "default_0.50",
            **default_metrics,
        }
    )

    locked_metrics = calculate_metrics(y_true, y_score, threshold=decision.threshold)
    rows.append(
        {
            "model": decision.model_name,
            "evaluation_split": "held_out_test",
            "operating_point": decision.operating_point,
            **locked_metrics,
        }
    )
    return pd.DataFrame(rows)


def build_final_predictions(
    y_true: pd.Series,
    y_score: np.ndarray,
    decision: FinalModelDecision,
) -> pd.DataFrame:
    """Build row-level held-out test predictions for auditability."""
    return pd.DataFrame(
        {
            "row_index_in_test_split": range(len(y_true)),
            "y_true": y_true.reset_index(drop=True).astype(int),
            "y_score": np.round(y_score, 8),
            "y_pred_default_0.50": (y_score >= 0.50).astype(int),
            f"y_pred_{decision.operating_point}": (y_score >= decision.threshold).astype(int),
        }
    )


def build_model_metadata(
    decision: FinalModelDecision,
    split: FeatureTargetSplit,
    model_path: Path,
) -> pd.DataFrame:
    """Create a compact metadata table for the serialized final model."""
    return pd.DataFrame(
        [
            {
                "artifact": "final_model",
                "model": decision.model_name,
                "best_params": json.dumps(decision.best_params, sort_keys=True),
                "locked_operating_point": decision.operating_point,
                "locked_threshold": decision.threshold,
                "seed": SEED,
                "train_rows": len(split.x_train),
                "test_rows": len(split.x_test),
                "model_path": _display_path(model_path),
            }
        ]
    )


def render_final_test_report(result: FinalizationResult) -> str:
    """Render the final held-out test evaluation report."""
    return f"""# Final Test Evaluation Report

## Scope

This report evaluates the locked final model once on the held-out 20% test set. Model selection, hyperparameter tuning, and threshold selection were completed before this evaluation.

## Locked Decision

- Model: {result.decision.model_name}
- Best parameters: `{json.dumps(result.decision.best_params, sort_keys=True)}`
- Locked operating point: {result.decision.operating_point}
- Locked threshold: {result.decision.threshold:.2f}
- Cross-validated recall before final test: {result.decision.cv_recall:.3f}
- Cross-validated ROC-AUC before final test: {result.decision.cv_roc_auc:.3f}
- Cross-validated F1 before final test: {result.decision.cv_f1:.3f}

## Split Summary

{result.split_summary.to_markdown(index=False)}

## Held-Out Test Metrics

{result.final_metrics.to_markdown(index=False)}

## Model Artifact

The fitted final pipeline is saved locally at `{_display_path(result.model_path)}`. This binary artifact is intentionally ignored by Git because it is generated output.

## Interpretation

The locked threshold should be interpreted as a recall-priority operating point. In this project context, false negatives are more serious than false positives because a false negative represents a heart-disease-positive patient predicted as negative. These results remain educational and should not be used for clinical decision-making.
"""


def render_model_card(result: FinalizationResult) -> str:
    """Render a concise model card for the final serialized pipeline."""
    locked_metrics = result.final_metrics.loc[
        result.final_metrics["operating_point"].eq(result.decision.operating_point)
    ].iloc[0]
    default_metrics = result.final_metrics.loc[
        result.final_metrics["operating_point"].eq("default_0.50")
    ].iloc[0]

    return f"""# Model Card - Heart Disease Risk Prediction

## Model Details

- Final model: {result.decision.model_name}
- Pipeline artifact: `{_display_path(result.model_path)}`
- Preprocessing: median imputation and scaling for numeric features, most-frequent imputation and one-hot encoding for nominal features, most-frequent imputation for `ca`, passthrough for binary features
- Random seed: {SEED}
- Locked threshold: {result.decision.threshold:.2f} ({result.decision.operating_point})

## Intended Use

This model is intended for an educational data mining project that demonstrates supervised classification, preprocessing, model comparison, threshold tuning, and interpretation on a small heart disease dataset.

## Not Intended Use

This model must not be used for diagnosis, triage, screening, treatment decisions, or patient-facing medical recommendations.

## Training Data

The model uses the processed Cleveland heart disease dataset supplied in `data/raw/heart.csv`. After duplicate removal, the dataset contains 302 rows. The final pipeline is fitted on the stratified training split with 241 rows and evaluated once on the held-out test split with 61 rows.

## Metrics

### Cross-Validated Training Metrics For Selected Model

- Recall: {result.decision.cv_recall:.3f}
- ROC-AUC: {result.decision.cv_roc_auc:.3f}
- F1: {result.decision.cv_f1:.3f}

### Held-Out Test Metrics

| operating_point | threshold | accuracy | precision | recall | specificity | f1 | roc_auc | average_precision | false_negative | false_positive |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| default_0.50 | {default_metrics["threshold"]:.2f} | {default_metrics["accuracy"]:.3f} | {default_metrics["precision"]:.3f} | {default_metrics["recall"]:.3f} | {default_metrics["specificity"]:.3f} | {default_metrics["f1"]:.3f} | {default_metrics["roc_auc"]:.3f} | {default_metrics["average_precision"]:.3f} | {int(default_metrics["false_negative"])} | {int(default_metrics["false_positive"])} |
| {result.decision.operating_point} | {locked_metrics["threshold"]:.2f} | {locked_metrics["accuracy"]:.3f} | {locked_metrics["precision"]:.3f} | {locked_metrics["recall"]:.3f} | {locked_metrics["specificity"]:.3f} | {locked_metrics["f1"]:.3f} | {locked_metrics["roc_auc"]:.3f} | {locked_metrics["average_precision"]:.3f} | {int(locked_metrics["false_negative"])} | {int(locked_metrics["false_positive"])} |

## Ethical And Practical Limitations

- The data is small, historical, and from a limited clinical context.
- The target is a simplified binary label derived from a richer disease-severity variable.
- Dataset-specific sex and age patterns should not be generalized to real populations.
- The model is not calibrated or validated for clinical deployment.
- Predictions are sensitive to the dataset's encoding, preprocessing assumptions, and threshold choice.

## Recommended Governance

Use this model only as a reproducible coursework artifact. Any real medical use would require modern representative data, clinical validation, calibration assessment, bias analysis, monitoring, and review by qualified medical professionals.
"""


def write_finalization_outputs(
    result: FinalizationResult,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Persist final model metrics, metadata, reports, and serialized pipeline."""
    results_dir.mkdir(parents=True, exist_ok=True)
    result.model_card_path.parent.mkdir(parents=True, exist_ok=True)
    result.final_test_report_path.parent.mkdir(parents=True, exist_ok=True)

    result.final_metrics.to_csv(results_dir / "final_test_metrics.csv", index=False)
    result.final_predictions.to_csv(results_dir / "final_test_predictions.csv", index=False)
    result.model_metadata.to_csv(results_dir / "final_model_metadata.csv", index=False)
    result.final_test_report_path.write_text(render_final_test_report(result), encoding="utf-8")
    result.model_card_path.write_text(render_model_card(result), encoding="utf-8")


def run_finalization(
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> FinalizationResult:
    """Fit the final model, evaluate the held-out test set once, and persist artifacts."""
    raw = load_raw_data()
    cleaned = clean_heart_data(raw).cleaned
    split = build_feature_target_split(cleaned)
    decision = load_final_decision(results_dir)
    pipeline = build_final_pipeline(decision)
    pipeline.fit(split.x_train, split.y_train)

    y_score = pipeline.predict_proba(split.x_test)[:, 1]
    final_metrics = build_final_test_metrics(split.y_test, y_score, decision)
    final_predictions = build_final_predictions(split.y_test, y_score, decision)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / FINAL_MODEL_PATH.name
    joblib.dump(pipeline, model_path)

    model_metadata = build_model_metadata(decision, split, model_path)
    result = FinalizationResult(
        decision=decision,
        split_summary=split.split.summary,
        final_metrics=final_metrics,
        final_predictions=final_predictions,
        model_metadata=model_metadata,
        model_path=model_path,
        model_card_path=reports_dir / MODEL_CARD_PATH.name,
        final_test_report_path=reports_dir / FINAL_TEST_REPORT_PATH.name,
    )
    write_finalization_outputs(result, results_dir)
    return result


if __name__ == "__main__":
    run_finalization()
