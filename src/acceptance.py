"""Final submission acceptance checklist."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.schema import RESULTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
SUBMISSION_CHECKLIST_REPORT_PATH = REPORTS_DIR / "SUBMISSION_CHECKLIST.md"

REQUIRED_ARTIFACTS = [
    ("raw_dataset", "data/raw/heart.csv", "Original raw dataset is present."),
    ("eda_notebook", "notebooks/01_eda.ipynb", "Narrative EDA notebook is present."),
    ("modelling_notebook", "notebooks/02_modelling.ipynb", "Narrative modelling notebook is present."),
    ("milestone_report", "reports/MILESTONE_1_REPORT.md", "Milestone report is present."),
    ("final_report", "reports/FINAL_PROJECT_REPORT.md", "Final project report is present."),
    ("model_card", "reports/MODEL_CARD.md", "Model card is present."),
    (
        "final_test_report",
        "reports/FINAL_TEST_EVALUATION_REPORT.md",
        "Final held-out test report is present.",
    ),
    (
        "reproducibility_report",
        "reports/REPRODUCIBILITY_CHECK_REPORT.md",
        "Reproducibility report is present.",
    ),
    ("final_metrics", "outputs/results/final_test_metrics.csv", "Final test metrics are present."),
    (
        "visualization_coverage",
        "outputs/results/visualization_requirements_coverage.csv",
        "Visualization coverage table is present.",
    ),
    (
        "final_model",
        "outputs/models/final_model.joblib",
        "Local generated final model artifact is present.",
    ),
]

SUCCESS_CRITERIA = [
    (
        "dataset_documented",
        "Dataset is documented, including sentinel values for `ca` and `thal`.",
        ["reports/DATA_DICTIONARY.md", "reports/DATA_CLEANING_REPORT.md"],
    ),
    (
        "leakage_controlled",
        "Preprocessing uses a shared pipeline fitted inside training folds.",
        ["reports/PREPROCESSING_PIPELINE_REPORT.md", "src/model_preprocess.py"],
    ),
    (
        "eda_quantified",
        "EDA includes plots plus statistical association tables.",
        ["reports/EXPLORATORY_DATA_ANALYSIS.md", "outputs/results/eda_chi_square_tests.csv"],
    ),
    (
        "outliers_documented",
        "Outliers are detected, documented, and preserved when plausible.",
        ["reports/OUTLIER_DETECTION_REPORT.md", "outputs/results/outlier_treatment_recommendations.csv"],
    ),
    (
        "models_compared",
        "Multiple classifiers and a dummy baseline are compared with stratified 5-fold CV.",
        ["reports/BASELINE_MODEL_COMPARISON.md", "outputs/results/cv_results.csv"],
    ),
    (
        "threshold_tuned",
        "Threshold tuning is documented with default and recall-priority operating points.",
        ["reports/MODEL_EVALUATION_REPORT.md", "outputs/results/evaluation_operating_points.csv"],
    ),
    (
        "test_evaluated_once",
        "Held-out test evaluation is reported after model and threshold decisions are locked.",
        ["reports/FINAL_TEST_EVALUATION_REPORT.md", "outputs/results/final_test_metrics.csv"],
    ),
    (
        "figures_exported",
        "Required figures are exported and indexed.",
        ["reports/VISUALIZATION_REPORT.md", "outputs/results/visualization_requirements_coverage.csv"],
    ),
    (
        "interpretability_completed",
        "Feature importance and medical insight analysis are documented.",
        ["reports/FEATURE_IMPORTANCE_REPORT.md", "outputs/results/interpret_insight_summary.csv"],
    ),
    (
        "reproducible",
        "The project can run end-to-end from raw data with fixed dependencies and seed.",
        ["reports/REPRODUCIBILITY_CHECK_REPORT.md", "src/run_all.py"],
    ),
]


@dataclass(frozen=True)
class AcceptanceResult:
    """Final acceptance-check outputs."""

    artifact_checklist: pd.DataFrame
    success_criteria: pd.DataFrame
    summary: pd.DataFrame
    report_path: Path


def build_artifact_checklist() -> pd.DataFrame:
    """Verify required submission artifacts exist."""
    rows = []
    for artifact, relative_path, description in REQUIRED_ARTIFACTS:
        path = PROJECT_ROOT / relative_path
        rows.append(
            {
                "artifact": artifact,
                "relative_path": relative_path,
                "description": description,
                "exists": path.exists(),
            }
        )
    return pd.DataFrame(rows)


def build_success_criteria_checklist() -> pd.DataFrame:
    """Map project success criteria to concrete generated artifacts."""
    rows = []
    for criterion, description, relative_paths in SUCCESS_CRITERIA:
        missing_paths = [path for path in relative_paths if not (PROJECT_ROOT / path).exists()]
        rows.append(
            {
                "criterion": criterion,
                "description": description,
                "evidence": "; ".join(relative_paths),
                "passed": not missing_paths,
                "missing_evidence": "; ".join(missing_paths),
            }
        )
    return pd.DataFrame(rows)


def build_acceptance_summary(
    artifacts: pd.DataFrame,
    criteria: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact pass/fail summary."""
    visualization = pd.read_csv(RESULTS_DIR / "visualization_requirements_coverage.csv")
    final_metrics = pd.read_csv(RESULTS_DIR / "final_test_metrics.csv")
    locked = final_metrics.loc[final_metrics["operating_point"].eq("recall_at_least_0.90")]
    locked = locked.iloc[0] if not locked.empty else final_metrics.tail(1).iloc[0]

    return pd.DataFrame(
        [
            {"metric": "required_artifacts_present", "value": bool(artifacts["exists"].all())},
            {"metric": "success_criteria_passed", "value": bool(criteria["passed"].all())},
            {
                "metric": "visualization_coverage",
                "value": f"{int(visualization['exists'].sum())}/{len(visualization)}",
            },
            {"metric": "final_model", "value": locked["model"]},
            {"metric": "locked_threshold", "value": locked["threshold"]},
            {"metric": "held_out_test_recall", "value": round(float(locked["recall"]), 3)},
            {"metric": "held_out_test_roc_auc", "value": round(float(locked["roc_auc"]), 3)},
            {"metric": "held_out_test_f1", "value": round(float(locked["f1"]), 3)},
        ]
    )


def render_acceptance_report(result: AcceptanceResult) -> str:
    """Render the final submission checklist report."""
    overall_status = (
        "passed"
        if bool(result.summary.loc[result.summary["metric"].eq("required_artifacts_present"), "value"].iloc[0])
        and bool(result.summary.loc[result.summary["metric"].eq("success_criteria_passed"), "value"].iloc[0])
        else "failed"
    )

    return f"""# Submission Checklist

## Scope

This checklist is the final acceptance pass for the Heart Disease Risk Prediction project. It maps the project requirements and success criteria to concrete repository artifacts.

## Overall Status

Submission readiness: **{overall_status}**

{result.summary.to_markdown(index=False)}

## Required Artifacts

{result.artifact_checklist.to_markdown(index=False)}

## Success Criteria Evidence

{result.success_criteria.to_markdown(index=False)}

## Submission Notes

- The final model artifact is generated locally at `outputs/models/final_model.joblib` and is intentionally ignored by Git.
- The primary report for grading is `reports/FINAL_PROJECT_REPORT.md`.
- The milestone report is `reports/MILESTONE_1_REPORT.md`.
- The ethical limitation is stated in the final report and model card: this project is educational, not clinical.
"""


def write_acceptance_outputs(result: AcceptanceResult) -> None:
    """Persist acceptance checklist outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.artifact_checklist.to_csv(RESULTS_DIR / "submission_artifact_checklist.csv", index=False)
    result.success_criteria.to_csv(RESULTS_DIR / "submission_success_criteria.csv", index=False)
    result.summary.to_csv(RESULTS_DIR / "submission_summary.csv", index=False)
    result.report_path.write_text(render_acceptance_report(result), encoding="utf-8")


def run_acceptance_check() -> AcceptanceResult:
    """Run the final submission acceptance check."""
    artifacts = build_artifact_checklist()
    criteria = build_success_criteria_checklist()
    summary = build_acceptance_summary(artifacts, criteria)
    result = AcceptanceResult(
        artifact_checklist=artifacts,
        success_criteria=criteria,
        summary=summary,
        report_path=SUBMISSION_CHECKLIST_REPORT_PATH,
    )
    write_acceptance_outputs(result)
    return result


if __name__ == "__main__":
    run_acceptance_check()
