# Submission Checklist

## Scope

This checklist is the final acceptance pass for the Heart Disease Risk Prediction project. It maps the project requirements and success criteria to concrete repository artifacts.

## Overall Status

Submission readiness: **passed**

| metric                     | value                  |
|:---------------------------|:-----------------------|
| required_artifacts_present | True                   |
| success_criteria_passed    | True                   |
| visualization_coverage     | 17/17                  |
| final_model                | Support Vector Machine |
| locked_threshold           | 0.4                    |
| held_out_test_recall       | 0.909                  |
| held_out_test_roc_auc      | 0.912                  |
| held_out_test_f1           | 0.811                  |

## Required Artifacts

| artifact               | relative_path                                           | description                                      | exists   |
|:-----------------------|:--------------------------------------------------------|:-------------------------------------------------|:---------|
| raw_dataset            | data/raw/heart.csv                                      | Original raw dataset is present.                 | True     |
| eda_notebook           | notebooks/01_eda.ipynb                                  | Narrative EDA notebook is present.               | True     |
| modelling_notebook     | notebooks/02_modelling.ipynb                            | Narrative modelling notebook is present.         | True     |
| milestone_report       | reports/MILESTONE_1_REPORT.md                           | Milestone report is present.                     | True     |
| final_report           | reports/FINAL_PROJECT_REPORT.md                         | Final project report is present.                 | True     |
| model_card             | reports/MODEL_CARD.md                                   | Model card is present.                           | True     |
| final_test_report      | reports/FINAL_TEST_EVALUATION_REPORT.md                 | Final held-out test report is present.           | True     |
| reproducibility_report | reports/REPRODUCIBILITY_CHECK_REPORT.md                 | Reproducibility report is present.               | True     |
| final_metrics          | outputs/results/final_test_metrics.csv                  | Final test metrics are present.                  | True     |
| visualization_coverage | outputs/results/visualization_requirements_coverage.csv | Visualization coverage table is present.         | True     |
| final_model            | outputs/models/final_model.joblib                       | Local generated final model artifact is present. | True     |

## Success Criteria Evidence

| criterion                  | description                                                                          | evidence                                                                                   | passed   | missing_evidence   |
|:---------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------|:---------|:-------------------|
| dataset_documented         | Dataset is documented, including sentinel values for `ca` and `thal`.                | reports/DATA_DICTIONARY.md; reports/DATA_CLEANING_REPORT.md                                | True     |                    |
| leakage_controlled         | Preprocessing uses a shared pipeline fitted inside training folds.                   | reports/PREPROCESSING_PIPELINE_REPORT.md; src/model_preprocess.py                          | True     |                    |
| eda_quantified             | EDA includes plots plus statistical association tables.                              | reports/EXPLORATORY_DATA_ANALYSIS.md; outputs/results/eda_chi_square_tests.csv             | True     |                    |
| outliers_documented        | Outliers are detected, documented, and preserved when plausible.                     | reports/OUTLIER_DETECTION_REPORT.md; outputs/results/outlier_treatment_recommendations.csv | True     |                    |
| models_compared            | Multiple classifiers and a dummy baseline are compared with stratified 5-fold CV.    | reports/BASELINE_MODEL_COMPARISON.md; outputs/results/cv_results.csv                       | True     |                    |
| threshold_tuned            | Threshold tuning is documented with default and recall-priority operating points.    | reports/MODEL_EVALUATION_REPORT.md; outputs/results/evaluation_operating_points.csv        | True     |                    |
| test_evaluated_once        | Held-out test evaluation is reported after model and threshold decisions are locked. | reports/FINAL_TEST_EVALUATION_REPORT.md; outputs/results/final_test_metrics.csv            | True     |                    |
| figures_exported           | Required figures are exported and indexed.                                           | reports/VISUALIZATION_REPORT.md; outputs/results/visualization_requirements_coverage.csv   | True     |                    |
| interpretability_completed | Feature importance and medical insight analysis are documented.                      | reports/FEATURE_IMPORTANCE_REPORT.md; outputs/results/interpret_insight_summary.csv        | True     |                    |
| reproducible               | The project can run end-to-end from raw data with fixed dependencies and seed.       | reports/REPRODUCIBILITY_CHECK_REPORT.md; src/run_all.py                                    | True     |                    |

## Submission Notes

- The final model artifact is generated locally at `outputs/models/final_model.joblib` and is intentionally ignored by Git.
- The primary report for grading is `reports/FINAL_PROJECT_REPORT.md`.
- The milestone report is `reports/MILESTONE_1_REPORT.md`.
- The ethical limitation is stated in the final report and model card: this project is educational, not clinical.
