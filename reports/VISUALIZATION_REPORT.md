# Visualization Report

## Scope

This report indexes all generated project figures, confirms required visualization coverage, and records image dimensions for documentation QA. All generated figures are PNG files intended for report insertion.

## Required Visualization Coverage

Covered required/optional visualizations: 17/17

| requirement                                  | figure                                                               | status            | relative_path                                                                        | exists   |
|:---------------------------------------------|:---------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------------|:---------|
| Target class distribution bar chart          | eda_target_distribution.png                                          | complete          | outputs/figures/eda_target_distribution.png                                          | True     |
| Age distribution histogram by target         | eda_age_distribution_by_target.png                                   | complete          | outputs/figures/eda_age_distribution_by_target.png                                   | True     |
| Heart disease percentage by age group        | eda_age_group_target_rate.png                                        | complete          | outputs/figures/eda_age_group_target_rate.png                                        | True     |
| Heart disease percentage by sex              | eda_sex_target_rate.png                                              | complete          | outputs/figures/eda_sex_target_rate.png                                              | True     |
| Pearson correlation heatmap                  | eda_pearson_correlation_heatmap.png                                  | complete          | outputs/figures/eda_pearson_correlation_heatmap.png                                  | True     |
| Box plots for outlier detection              | outliers_numeric_boxplots.png                                        | complete          | outputs/figures/outliers_numeric_boxplots.png                                        | True     |
| Chest pain type vs target count plot         | eda_chest_pain_vs_target.png                                         | complete          | outputs/figures/eda_chest_pain_vs_target.png                                         | True     |
| Exercise-induced angina vs target count plot | eda_exang_vs_target.png                                              | complete          | outputs/figures/eda_exang_vs_target.png                                              | True     |
| Maximum heart rate vs target box plot        | eda_thalach_by_target.png                                            | complete          | outputs/figures/eda_thalach_by_target.png                                            | True     |
| Cholesterol vs target box plot               | eda_cholesterol_by_target.png                                        | complete          | outputs/figures/eda_cholesterol_by_target.png                                        | True     |
| Mutual information feature relevance chart   | eda_mutual_information.png                                           | complete          | outputs/figures/eda_mutual_information.png                                           | True     |
| Model CV-score comparison with error bars    | visualization_cv_score_comparison_error_bars.png                     | complete          | outputs/figures/visualization_cv_score_comparison_error_bars.png                     | True     |
| ROC curve overlay                            | evaluation_roc_curves.png                                            | complete          | outputs/figures/evaluation_roc_curves.png                                            | True     |
| Precision-recall curve overlay               | evaluation_precision_recall_curves.png                               | complete          | outputs/figures/evaluation_precision_recall_curves.png                               | True     |
| Confusion matrix heatmaps for top models     | evaluation_confusion_support_vector_machine_recall_at_least_0.90.png | complete          | outputs/figures/evaluation_confusion_support_vector_machine_recall_at_least_0.90.png | True     |
| Feature importance chart for final candidate | interpret_permutation_importance.png                                 | complete          | outputs/figures/interpret_permutation_importance.png                                 | True     |
| Calibration plot for final candidate         | visualization_svm_calibration_curve.png                              | complete_optional | outputs/figures/visualization_svm_calibration_curve.png                              | True     |

## Figure Manifest

| figure                                                               | relative_path                                                                        |   width_px |   height_px |   file_size_bytes |
|:---------------------------------------------------------------------|:-------------------------------------------------------------------------------------|-----------:|------------:|------------------:|
| eda_age_distribution_by_target.png                                   | outputs/figures/eda_age_distribution_by_target.png                                   |       2053 |        1154 |            144542 |
| eda_age_group_target_rate.png                                        | outputs/figures/eda_age_group_target_rate.png                                        |       2054 |        1155 |             79226 |
| eda_chest_pain_vs_target.png                                         | outputs/figures/eda_chest_pain_vs_target.png                                         |       2652 |        1155 |            135621 |
| eda_cholesterol_by_target.png                                        | outputs/figures/eda_cholesterol_by_target.png                                        |       1753 |        1154 |             75805 |
| eda_exang_vs_target.png                                              | outputs/figures/eda_exang_vs_target.png                                              |       2052 |        1155 |             96459 |
| eda_mutual_information.png                                           | outputs/figures/eda_mutual_information.png                                           |       2052 |        1454 |             85906 |
| eda_pearson_correlation_heatmap.png                                  | outputs/figures/eda_pearson_correlation_heatmap.png                                  |       1959 |        1453 |            182559 |
| eda_sex_target_rate.png                                              | outputs/figures/eda_sex_target_rate.png                                              |       1754 |        1154 |             65428 |
| eda_target_distribution.png                                          | outputs/figures/eda_target_distribution.png                                          |       1752 |        1154 |             63520 |
| eda_thalach_by_target.png                                            | outputs/figures/eda_thalach_by_target.png                                            |       1752 |        1154 |             71468 |
| evaluation_confusion_support_vector_machine_default_0.50.png         | outputs/figures/evaluation_confusion_support_vector_machine_default_0.50.png         |       1287 |        1153 |             56180 |
| evaluation_confusion_support_vector_machine_recall_at_least_0.90.png | outputs/figures/evaluation_confusion_support_vector_machine_recall_at_least_0.90.png |       1287 |        1153 |             61095 |
| evaluation_confusion_xgboost_default_0.50.png                        | outputs/figures/evaluation_confusion_xgboost_default_0.50.png                        |       1287 |        1153 |             52310 |
| evaluation_confusion_xgboost_recall_at_least_0.90.png                | outputs/figures/evaluation_confusion_xgboost_recall_at_least_0.90.png                |       1287 |        1153 |             56282 |
| evaluation_oof_metric_comparison.png                                 | outputs/figures/evaluation_oof_metric_comparison.png                                 |       3252 |        1454 |            249872 |
| evaluation_precision_recall_curves.png                               | outputs/figures/evaluation_precision_recall_curves.png                               |       2052 |        1454 |            304714 |
| evaluation_roc_curves.png                                            | outputs/figures/evaluation_roc_curves.png                                            |       2052 |        1454 |            255254 |
| evaluation_threshold_sweep.png                                       | outputs/figures/evaluation_threshold_sweep.png                                       |       2652 |        1454 |            275160 |
| interpret_logistic_coefficients.png                                  | outputs/figures/interpret_logistic_coefficients.png                                  |       2352 |        1754 |            135087 |
| interpret_permutation_importance.png                                 | outputs/figures/interpret_permutation_importance.png                                 |       2352 |        1754 |             96861 |
| interpret_svm_coefficients.png                                       | outputs/figures/interpret_svm_coefficients.png                                       |       2350 |        1754 |            130889 |
| interpret_xgboost_importances.png                                    | outputs/figures/interpret_xgboost_importances.png                                    |       2353 |        1754 |            106057 |
| outliers_age_histogram_kde.png                                       | outputs/figures/outliers_age_histogram_kde.png                                       |       2053 |        1154 |            135103 |
| outliers_boxplots_by_target.png                                      | outputs/figures/outliers_boxplots_by_target.png                                      |       5352 |        1154 |            176830 |
| outliers_chol_histogram_kde.png                                      | outputs/figures/outliers_chol_histogram_kde.png                                      |       2052 |        1154 |            152181 |
| outliers_numeric_boxplots.png                                        | outputs/figures/outliers_numeric_boxplots.png                                        |       2352 |        3554 |            164781 |
| outliers_oldpeak_histogram_kde.png                                   | outputs/figures/outliers_oldpeak_histogram_kde.png                                   |       2052 |        1154 |            107754 |
| outliers_thalach_histogram_kde.png                                   | outputs/figures/outliers_thalach_histogram_kde.png                                   |       2053 |        1154 |            139791 |
| outliers_trestbps_histogram_kde.png                                  | outputs/figures/outliers_trestbps_histogram_kde.png                                  |       2053 |        1154 |            140014 |
| tuning_model_comparison.png                                          | outputs/figures/tuning_model_comparison.png                                          |       2352 |        1454 |            150570 |
| visualization_cv_score_comparison_error_bars.png                     | outputs/figures/visualization_cv_score_comparison_error_bars.png                     |       3252 |        1454 |            241512 |
| visualization_svm_calibration_curve.png                              | outputs/figures/visualization_svm_calibration_curve.png                              |       1752 |        1454 |            136755 |

## Notes

- Core EDA plots, outlier diagnostics, model comparison figures, ROC/PR curves, confusion matrices, and feature-importance plots are available under `outputs/figures/`.
- `visualization_cv_score_comparison_error_bars.png` provides the explicit +/- 1 standard deviation CV comparison required for model reporting.
- `visualization_svm_calibration_curve.png` is optional but useful for discussing probability calibration before final threshold selection.
