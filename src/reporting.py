"""Generate consolidated project documentation reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.schema import RESULTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MILESTONE_REPORT_PATH = REPORTS_DIR / "MILESTONE_1_REPORT.md"
FINAL_REPORT_PATH = REPORTS_DIR / "FINAL_PROJECT_REPORT.md"


@dataclass(frozen=True)
class ReportingResult:
    """Generated project report paths."""

    milestone_report_path: Path
    final_report_path: Path


def _read_table(results_dir: Path, filename: str) -> pd.DataFrame:
    table = pd.read_csv(results_dir / filename)
    if "Unnamed: 0" in table.columns:
        table = table.rename(columns={"Unnamed: 0": "feature"})
    return table


def _markdown_table(dataframe: pd.DataFrame, max_rows: int | None = None) -> str:
    table = dataframe.copy()
    if max_rows is not None:
        table = table.head(max_rows)
    return table.to_markdown(index=False)


def _metric_value(summary: pd.DataFrame, metric: str) -> str:
    value = summary.loc[summary["metric"].eq(metric), "value"].iloc[0]
    return str(value)


def _figure(filename: str, caption: str) -> str:
    return f"![{caption}](../outputs/figures/{filename})"


def render_milestone_report(results_dir: Path = RESULTS_DIR) -> str:
    """Render the milestone 1 report from generated analysis artifacts."""
    cleaning = _read_table(results_dir, "cleaning_summary.csv")
    sentinel = _read_table(results_dir, "sentinel_replacements.csv")
    ranges = _read_table(results_dir, "numeric_range_validation.csv")
    age_groups = _read_table(results_dir, "eda_age_group_analysis.csv")
    sex_analysis = _read_table(results_dir, "eda_sex_analysis.csv")
    chi_square = _read_table(results_dir, "eda_chi_square_tests.csv")
    mann_whitney = _read_table(results_dir, "eda_mann_whitney_tests.csv")
    mutual_info = _read_table(results_dir, "eda_mutual_information.csv")
    pearson = _read_table(results_dir, "eda_pearson_correlation.csv")
    iqr_summary = _read_table(results_dir, "outlier_iqr_summary.csv")
    zscore_summary = _read_table(results_dir, "outlier_zscore_summary.csv")
    outlier_treatment = _read_table(results_dir, "outlier_treatment_recommendations.csv")
    visualization_coverage = _read_table(results_dir, "visualization_requirements_coverage.csv")

    younger_rate = age_groups.loc[age_groups["age_group"].eq("<40"), "heart_disease_percentage"].iloc[0]
    female_rate = sex_analysis.loc[sex_analysis["sex_label"].eq("Female"), "heart_disease_percentage"].iloc[0]
    male_rate = sex_analysis.loc[sex_analysis["sex_label"].eq("Male"), "heart_disease_percentage"].iloc[0]

    return f"""# Milestone 1 Report - Heart Disease Risk Prediction

## Project Scope

This milestone documents the data-understanding, cleaning, outlier-detection, correlation-analysis, and visual-assessment phases of the Heart Disease Risk Prediction data mining project. The work uses the processed Cleveland heart disease dataset supplied as `data/raw/heart.csv`.

This project is educational and analytical. It is not a clinical diagnostic system, screening tool, or medical recommendation engine.

## Dataset Overview

- Raw rows: {_metric_value(cleaning, "raw_rows")}
- Raw columns: {_metric_value(cleaning, "raw_columns")}
- Cleaned rows after duplicate removal: {_metric_value(cleaning, "cleaned_rows")}
- Explicit raw missing values: {_metric_value(cleaning, "raw_explicit_missing_values")}
- Missing values after sentinel decoding: {_metric_value(cleaning, "cleaned_missing_values")}
- Positive class count after cleaning: {_metric_value(cleaning, "target_1_count_after_cleaning")}
- Negative class count after cleaning: {_metric_value(cleaning, "target_0_count_after_cleaning")}

## Data Cleaning Summary

Cleaning was intentionally conservative. The duplicate row was removed before train/test splitting to prevent leakage. Sentinel-encoded missing values were decoded to `NaN` and left for the modelling pipeline to impute inside cross-validation folds.

{_markdown_table(cleaning)}

### Sentinel Decoding

{_markdown_table(sentinel)}

### Numeric Range Review

{_markdown_table(ranges)}

The cholesterol maximum is extreme but biologically plausible, so it is documented rather than removed. Resting blood pressure and `oldpeak` outliers are also preserved because the values are plausible clinical observations.

## Exploratory Visual Assessment

### Target And Demographic Patterns

The younger `<40` group has a heart-disease-positive rate of {younger_rate:.2f}% in the training split. Female patients show a {female_rate:.2f}% positive rate and male patients show a {male_rate:.2f}% positive rate in this dataset. These are dataset-specific observations, not population-level medical claims.

{_markdown_table(age_groups)}

{_markdown_table(sex_analysis)}

{_figure("eda_target_distribution.png", "Target class distribution")}

{_figure("eda_age_group_target_rate.png", "Heart disease percentage by age group")}

{_figure("eda_sex_target_rate.png", "Heart disease percentage by sex")}

### Correlation And Association Analysis

The analysis goes beyond visual inspection by using Pearson/Spearman correlations, chi-square tests for categorical predictors, Mann-Whitney tests for numerical predictors, and mutual information scores.

#### Pearson Correlation Matrix

{_markdown_table(pearson)}

#### Chi-Square Tests

{_markdown_table(chi_square)}

#### Mann-Whitney Tests

{_markdown_table(mann_whitney)}

#### Mutual Information Scores

{_markdown_table(mutual_info)}

{_figure("eda_pearson_correlation_heatmap.png", "Pearson correlation heatmap")}

{_figure("eda_mutual_information.png", "Mutual information feature relevance")}

## Outlier Detection

Outliers were reviewed with both visual plots and statistical rules. No records were removed because the detected observations are plausible for a medical dataset and may contain useful predictive signal.

### IQR Summary

{_markdown_table(iqr_summary)}

### Z-Score Summary

{_markdown_table(zscore_summary)}

### Treatment Recommendation

{_markdown_table(outlier_treatment)}

{_figure("outliers_numeric_boxplots.png", "Numeric box plots for outlier detection")}

## Visualization Coverage

All required visualization categories are covered by exported PNG files under `outputs/figures/`.

{_markdown_table(visualization_coverage)}

## Milestone 1 Conclusion

The dataset is small, mostly clean, and mildly imbalanced. The main cleaning risk is not explicit missingness but sentinel-encoded missing values in `ca` and `thal`. The strongest early analytical signals come from chest-pain type, thalassemia category, exercise-induced angina, vessel count, `oldpeak`, and maximum heart rate. Outliers are documented and preserved for modelling.
"""


def render_final_report(results_dir: Path = RESULTS_DIR) -> str:
    """Render the consolidated final project report from generated artifacts."""
    cleaning = _read_table(results_dir, "cleaning_summary.csv")
    split_summary = _read_table(results_dir, "preprocess_split_summary.csv")
    column_routing = _read_table(results_dir, "preprocess_column_routing.csv")
    model_inventory = _read_table(results_dir, "baseline_model_inventory.csv")
    baseline = _read_table(results_dir, "baseline_model_comparison.csv")
    tuned = _read_table(results_dir, "tuned_model_comparison.csv")
    selected = _read_table(results_dir, "tuned_selected_model.csv")
    oof_metrics = _read_table(results_dir, "evaluation_oof_metrics.csv")
    operating_points = _read_table(results_dir, "evaluation_operating_points.csv")
    insight_summary = _read_table(results_dir, "interpret_insight_summary.csv")
    permutation = _read_table(results_dir, "interpret_permutation_importance.csv")
    visualization_manifest = _read_table(results_dir, "visualization_figure_manifest.csv")

    top_baseline = baseline.sort_values(["recall_mean", "roc_auc_mean", "f1_mean"], ascending=False).iloc[0]
    selected_model = selected["model"].iloc[0]
    selected_params = selected["best_params"].iloc[0]
    tuned_selected = tuned.loc[tuned["model"].eq(selected_model)].iloc[0]
    svm_operating_points = operating_points.loc[
        operating_points["model"].eq("Support Vector Machine")
    ]

    return f"""# Final Project Report - Heart Disease Risk Prediction

## Project Name

Heart Disease Risk Prediction Using Data Mining and Machine Learning

## Project Format

Single-author educational data mining project. The implementation is structured as reusable Python modules under `src/`, with generated reports, figures, and result tables stored under `reports/` and `outputs/`.

## Problem Statement

The goal is to predict whether a patient has heart disease using clinical attributes such as age, sex, chest-pain type, cholesterol, resting blood pressure, maximum heart rate, exercise-induced angina, ST depression, vessel count, and thalassemia category. The project compares multiple supervised classifiers and studies which variables appear most influential.

This work is not a diagnostic product. The Cleveland dataset is small, historical, and not representative of modern clinical populations.

## Dataset Description

- Source file: `data/raw/heart.csv`
- Raw records: {_metric_value(cleaning, "raw_rows")}
- Cleaned records: {_metric_value(cleaning, "cleaned_rows")}
- Features: 13 predictors plus one binary target
- Target definition: `target = 1` indicates heart disease presence; `target = 0` indicates absence
- Main cleaning issue: `ca == 4` and `thal == 0` are sentinel-encoded missing values

## Tools And Libraries

The project uses Python 3.11+, pandas, NumPy, SciPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Joblib, Jupyter tooling, and Pytest. Dependencies are pinned in `requirements.txt`.

## Reproducible Workflow

1. Load and audit the raw dataset.
2. Build the data dictionary and validate categorical encodings.
3. Remove the duplicate row and decode sentinel values.
4. Lock a stratified 80/20 train/test split.
5. Run EDA, statistical association tests, and outlier review on the training split.
6. Build one shared leak-proof preprocessing pipeline.
7. Compare classifiers with stratified 5-fold cross-validation.
8. Tune the strongest candidates.
9. Evaluate out-of-fold predictions and tune thresholds.
10. Interpret the selected model with coefficients, tree importances, and permutation importance.
11. Export all figures and reports.

## Preprocessing Workflow

The preprocessing workflow is implemented as a shared `ColumnTransformer` inside every model pipeline. Imputation, scaling, and encoding are fitted inside training folds only, preventing leakage.

### Split Summary

{_markdown_table(split_summary)}

### Column Routing

{_markdown_table(column_routing)}

## Classification Models

The project compares a dummy baseline against linear, distance-based, kernel, tree, ensemble, probabilistic, boosting, and gradient-boosted tabular models.

{_markdown_table(model_inventory)}

## Baseline Cross-Validation Results

Baseline model selection uses stratified 5-fold cross-validation on the training split. The top baseline by recall is {top_baseline["model"]} with recall {top_baseline["recall_mean"]:.3f}, ROC-AUC {top_baseline["roc_auc_mean"]:.3f}, and F1 {top_baseline["f1_mean"]:.3f}.

{_markdown_table(baseline)}

{_figure("visualization_cv_score_comparison_error_bars.png", "Baseline CV comparison with standard deviation error bars")}

## Hyperparameter Tuning Results

The tuned shortlist contains Logistic Regression, Support Vector Machine, and XGBoost. The selected tuned candidate is {selected_model}.

- Selected parameters: `{selected_params}`
- Tuned recall: {tuned_selected["recall_mean"]:.3f}
- Tuned ROC-AUC: {tuned_selected["roc_auc_mean"]:.3f}
- Tuned F1: {tuned_selected["f1_mean"]:.3f}

{_markdown_table(tuned)}

{_figure("tuning_model_comparison.png", "Tuned model comparison")}

## Evaluation Metrics And Threshold Analysis

Because this is a medical screening-style problem, recall is prioritized over accuracy. A false negative means a heart-disease-positive patient is missed; a false positive means a patient may receive unnecessary follow-up.

The table below uses out-of-fold training predictions only. The held-out test set remains locked at this documentation stage and should be evaluated once during final packaging/model-card generation.

{_markdown_table(oof_metrics)}

### Selected Model Operating Points

{_markdown_table(svm_operating_points)}

{_figure("evaluation_roc_curves.png", "ROC curve overlay")}

{_figure("evaluation_precision_recall_curves.png", "Precision-recall curve overlay")}

{_figure("evaluation_confusion_support_vector_machine_recall_at_least_0.90.png", "Selected SVM tuned-threshold confusion matrix")}

## Feature Importance And Observations

The strongest recurring predictors are thalassemia category, chest-pain type, number of major vessels, exercise-induced angina, `oldpeak`, maximum heart rate, sex, and slope. These are predictive associations in this dataset, not causal medical conclusions.

{_markdown_table(insight_summary)}

### Final Candidate Permutation Importance

{_markdown_table(permutation)}

{_figure("interpret_permutation_importance.png", "Permutation importance for the selected model")}

## Visual Deliverables

The project exports 300 dpi PNG figures for EDA, outlier diagnostics, model evaluation, tuning, interpretation, and visualization QA.

{_markdown_table(visualization_manifest)}

## Conclusions

- The dataset is suitable for a compact, professional supervised classification study but is too small and historical for clinical deployment.
- Sentinel handling for `ca` and `thal` is the most important cleaning decision.
- The tuned linear Support Vector Machine is the current selected model because it provides the strongest recall-oriented cross-validation profile.
- A threshold of 0.40 on out-of-fold SVM predictions reaches recall above 0.90 while preserving a reasonable F1 score.
- Feature interpretation consistently highlights chest-pain type, thalassemia category, vessel count, exercise-induced angina, `oldpeak`, and maximum heart rate.

## Limitations

- The dataset contains only 303 raw records before duplicate removal.
- The data comes from a historical Cleveland Clinic subset and does not represent modern populations.
- The binary target simplifies the original heart-disease severity scale.
- Final held-out test-set evaluation and model-card packaging are intentionally deferred until the final quality step.
- The model should never be used as a substitute for professional medical diagnosis.
"""


def run_reporting(
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> ReportingResult:
    """Write milestone and final Markdown reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    milestone_path = reports_dir / MILESTONE_REPORT_PATH.name
    final_path = reports_dir / FINAL_REPORT_PATH.name
    milestone_path.write_text(render_milestone_report(results_dir), encoding="utf-8")
    final_path.write_text(render_final_report(results_dir), encoding="utf-8")
    return ReportingResult(
        milestone_report_path=milestone_path,
        final_report_path=final_path,
    )


if __name__ == "__main__":
    run_reporting()
