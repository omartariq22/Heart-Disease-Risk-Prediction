# Model Evaluation Report

## Scope

This evaluation uses out-of-fold predictions from stratified 5-fold cross-validation on the training split only. The held-out test set is intentionally not evaluated in Step 9 because hyperparameter tuning has not been completed yet.

## Split Summary

| split   |   rows |   target_0_count |   target_1_count |   target_1_percentage |
|:--------|-------:|-----------------:|-----------------:|----------------------:|
| train   |    241 |              110 |              131 |                 54.36 |
| test    |     61 |               28 |               33 |                 54.1  |

## Baseline Cross-Validation Ranking

| model                  |   accuracy_mean |   accuracy_std |   precision_mean |   precision_std |   recall_mean |   recall_std |   f1_mean |   f1_std |   roc_auc_mean |   roc_auc_std |   average_precision_mean |   average_precision_std |
|:-----------------------|----------------:|---------------:|-----------------:|----------------:|--------------:|-------------:|----------:|---------:|---------------:|--------------:|-------------------------:|------------------------:|
| XGBoost                |        0.834099 |       0.032375 |         0.825048 |        0.044996 |      0.88547  |     0.027263 |  0.853404 | 0.024995 |       0.904856 |      0.032022 |                 0.919957 |                0.021075 |
| Support Vector Machine |        0.834014 |       0.039005 |         0.824594 |        0.047532 |      0.885185 |     0.028214 |  0.853292 | 0.032072 |       0.908638 |      0.046205 |                 0.910691 |                0.053101 |
| Gaussian Naive Bayes   |        0.813691 |       0.089127 |         0.812465 |        0.099306 |      0.870085 |     0.064642 |  0.837674 | 0.069837 |       0.878982 |      0.051329 |                 0.901076 |                0.038239 |
| Logistic Regression    |        0.842177 |       0.076265 |         0.849337 |        0.086084 |      0.8698   |     0.044737 |  0.858436 | 0.062333 |       0.921717 |      0.038504 |                 0.934301 |                0.031298 |
| K-Nearest Neighbors    |        0.821599 |       0.037673 |         0.822547 |        0.049388 |      0.862108 |     0.065048 |  0.839766 | 0.034073 |       0.886299 |      0.036374 |                 0.864116 |                0.040776 |
| Random Forest          |        0.817177 |       0.058461 |         0.821258 |        0.067313 |      0.854416 |     0.051138 |  0.836081 | 0.047891 |       0.905795 |      0.043056 |                 0.918996 |                0.028793 |
| Gradient Boosting      |        0.796599 |       0.050088 |         0.799866 |        0.069602 |      0.847293 |     0.038546 |  0.820388 | 0.032214 |       0.881507 |      0.034852 |                 0.906082 |                0.024033 |
| Decision Tree          |        0.76352  |       0.059798 |         0.773872 |        0.059393 |      0.801994 |     0.06057  |  0.786732 | 0.052241 |       0.760088 |      0.061047 |                 0.729795 |                0.057412 |
| Dummy Stratified       |        0.506122 |       0.01369  |         0.54     |        0.014907 |      0.618234 |     0.006371 |  0.576441 | 0.011208 |       0.49548  |      0.013349 |                 0.541405 |                0.010882 |

## Out-of-Fold Metrics At Default Threshold 0.50

| model                  |   threshold |   accuracy |   precision |   recall |   specificity |       f1 |   roc_auc |   average_precision |   true_negative |   false_positive |   false_negative |   true_positive |
|:-----------------------|------------:|-----------:|------------:|---------:|--------------:|---------:|----------:|--------------------:|----------------:|-----------------:|-----------------:|----------------:|
| XGBoost                |         0.5 |   0.834025 |    0.822695 | 0.885496 |      0.772727 | 0.852941 |  0.901388 |            0.91253  |              85 |               25 |               15 |             116 |
| Support Vector Machine |         0.5 |   0.834025 |    0.827338 | 0.877863 |      0.781818 | 0.851852 |  0.899931 |            0.894299 |              86 |               24 |               16 |             115 |
| Logistic Regression    |         0.5 |   0.842324 |    0.844444 | 0.870229 |      0.809091 | 0.857143 |  0.912422 |            0.918929 |              89 |               21 |               17 |             114 |
| Gaussian Naive Bayes   |         0.5 |   0.813278 |    0.802817 | 0.870229 |      0.745455 | 0.835165 |  0.857946 |            0.880244 |              82 |               28 |               17 |             114 |
| Random Forest          |         0.5 |   0.817427 |    0.81295  | 0.862595 |      0.763636 | 0.837037 |  0.899549 |            0.906791 |              84 |               26 |               18 |             113 |
| K-Nearest Neighbors    |         0.5 |   0.821577 |    0.818841 | 0.862595 |      0.772727 | 0.840149 |  0.881541 |            0.856275 |              85 |               25 |               18 |             113 |
| Gradient Boosting      |         0.5 |   0.79668  |    0.792857 | 0.847328 |      0.736364 | 0.819188 |  0.867939 |            0.885001 |              81 |               29 |               20 |             111 |
| Decision Tree          |         0.5 |   0.763485 |    0.772059 | 0.801527 |      0.718182 | 0.786517 |  0.759854 |            0.72671  |              79 |               31 |               26 |             105 |
| Dummy Stratified       |         0.5 |   0.506224 |    0.54     | 0.618321 |      0.372727 | 0.576512 |  0.495524 |            0.541362 |              41 |               69 |               50 |              81 |

## Threshold Sweep Operating Points

| model                  | operating_point      |   threshold |   accuracy |   precision |   recall |   specificity |       f1 |   roc_auc |   average_precision |   true_negative |   false_positive |   false_negative |   true_positive |
|:-----------------------|:---------------------|------------:|-----------:|------------:|---------:|--------------:|---------:|----------:|--------------------:|----------------:|-----------------:|-----------------:|----------------:|
| XGBoost                | default_0.50         |        0.5  |   0.834025 |    0.822695 | 0.885496 |      0.772727 | 0.852941 |  0.901388 |            0.91253  |              85 |               25 |               15 |             116 |
| Support Vector Machine | default_0.50         |        0.5  |   0.834025 |    0.827338 | 0.877863 |      0.781818 | 0.851852 |  0.899931 |            0.894299 |              86 |               24 |               16 |             115 |
| Logistic Regression    | default_0.50         |        0.5  |   0.842324 |    0.844444 | 0.870229 |      0.809091 | 0.857143 |  0.912422 |            0.918929 |              89 |               21 |               17 |             114 |
| Gaussian Naive Bayes   | default_0.50         |        0.5  |   0.813278 |    0.802817 | 0.870229 |      0.745455 | 0.835165 |  0.857946 |            0.880244 |              82 |               28 |               17 |             114 |
| Random Forest          | default_0.50         |        0.5  |   0.817427 |    0.81295  | 0.862595 |      0.763636 | 0.837037 |  0.899549 |            0.906791 |              84 |               26 |               18 |             113 |
| K-Nearest Neighbors    | default_0.50         |        0.5  |   0.821577 |    0.818841 | 0.862595 |      0.772727 | 0.840149 |  0.881541 |            0.856275 |              85 |               25 |               18 |             113 |
| Gradient Boosting      | default_0.50         |        0.5  |   0.79668  |    0.792857 | 0.847328 |      0.736364 | 0.819188 |  0.867939 |            0.885001 |              81 |               29 |               20 |             111 |
| Decision Tree          | default_0.50         |        0.5  |   0.763485 |    0.772059 | 0.801527 |      0.718182 | 0.786517 |  0.759854 |            0.72671  |              79 |               31 |               26 |             105 |
| Dummy Stratified       | default_0.50         |        0.5  |   0.506224 |    0.54     | 0.618321 |      0.372727 | 0.576512 |  0.495524 |            0.541362 |              41 |               69 |               50 |              81 |
| Support Vector Machine | max_f1               |        0.4  |   0.838174 |    0.802632 | 0.931298 |      0.727273 | 0.862191 |  0.899931 |            0.894299 |              80 |               30 |                9 |             122 |
| Support Vector Machine | recall_at_least_0.90 |        0.4  |   0.838174 |    0.802632 | 0.931298 |      0.727273 | 0.862191 |  0.899931 |            0.894299 |              80 |               30 |                9 |             122 |
| XGBoost                | max_f1               |        0.53 |   0.842324 |    0.834532 | 0.885496 |      0.790909 | 0.859259 |  0.901388 |            0.91253  |              87 |               23 |               15 |             116 |
| XGBoost                | recall_at_least_0.90 |        0.25 |   0.821577 |    0.782051 | 0.931298 |      0.690909 | 0.850174 |  0.901388 |            0.91253  |              76 |               34 |                9 |             122 |

## Tuned Recall Operating Points

| model                  | operating_point      |   threshold |   accuracy |   precision |   recall |   specificity |       f1 |   roc_auc |   average_precision |   true_negative |   false_positive |   false_negative |   true_positive |
|:-----------------------|:---------------------|------------:|-----------:|------------:|---------:|--------------:|---------:|----------:|--------------------:|----------------:|-----------------:|-----------------:|----------------:|
| Support Vector Machine | recall_at_least_0.90 |        0.4  |   0.838174 |    0.802632 | 0.931298 |      0.727273 | 0.862191 |  0.899931 |            0.894299 |              80 |               30 |                9 |             122 |
| XGBoost                | recall_at_least_0.90 |        0.25 |   0.821577 |    0.782051 | 0.931298 |      0.690909 | 0.850174 |  0.901388 |            0.91253  |              76 |               34 |                9 |             122 |

## Confusion Matrix Counts

| model                  | operating_point      |   actual |   predicted |   count |
|:-----------------------|:---------------------|---------:|------------:|--------:|
| XGBoost                | default_0.50         |        0 |           0 |      85 |
| XGBoost                | default_0.50         |        0 |           1 |      25 |
| XGBoost                | default_0.50         |        1 |           0 |      15 |
| XGBoost                | default_0.50         |        1 |           1 |     116 |
| Support Vector Machine | default_0.50         |        0 |           0 |      86 |
| Support Vector Machine | default_0.50         |        0 |           1 |      24 |
| Support Vector Machine | default_0.50         |        1 |           0 |      16 |
| Support Vector Machine | default_0.50         |        1 |           1 |     115 |
| Logistic Regression    | default_0.50         |        0 |           0 |      89 |
| Logistic Regression    | default_0.50         |        0 |           1 |      21 |
| Logistic Regression    | default_0.50         |        1 |           0 |      17 |
| Logistic Regression    | default_0.50         |        1 |           1 |     114 |
| Gaussian Naive Bayes   | default_0.50         |        0 |           0 |      82 |
| Gaussian Naive Bayes   | default_0.50         |        0 |           1 |      28 |
| Gaussian Naive Bayes   | default_0.50         |        1 |           0 |      17 |
| Gaussian Naive Bayes   | default_0.50         |        1 |           1 |     114 |
| Random Forest          | default_0.50         |        0 |           0 |      84 |
| Random Forest          | default_0.50         |        0 |           1 |      26 |
| Random Forest          | default_0.50         |        1 |           0 |      18 |
| Random Forest          | default_0.50         |        1 |           1 |     113 |
| K-Nearest Neighbors    | default_0.50         |        0 |           0 |      85 |
| K-Nearest Neighbors    | default_0.50         |        0 |           1 |      25 |
| K-Nearest Neighbors    | default_0.50         |        1 |           0 |      18 |
| K-Nearest Neighbors    | default_0.50         |        1 |           1 |     113 |
| Gradient Boosting      | default_0.50         |        0 |           0 |      81 |
| Gradient Boosting      | default_0.50         |        0 |           1 |      29 |
| Gradient Boosting      | default_0.50         |        1 |           0 |      20 |
| Gradient Boosting      | default_0.50         |        1 |           1 |     111 |
| Decision Tree          | default_0.50         |        0 |           0 |      79 |
| Decision Tree          | default_0.50         |        0 |           1 |      31 |
| Decision Tree          | default_0.50         |        1 |           0 |      26 |
| Decision Tree          | default_0.50         |        1 |           1 |     105 |
| Dummy Stratified       | default_0.50         |        0 |           0 |      41 |
| Dummy Stratified       | default_0.50         |        0 |           1 |      69 |
| Dummy Stratified       | default_0.50         |        1 |           0 |      50 |
| Dummy Stratified       | default_0.50         |        1 |           1 |      81 |
| Support Vector Machine | max_f1               |        0 |           0 |      80 |
| Support Vector Machine | max_f1               |        0 |           1 |      30 |
| Support Vector Machine | max_f1               |        1 |           0 |       9 |
| Support Vector Machine | max_f1               |        1 |           1 |     122 |
| Support Vector Machine | recall_at_least_0.90 |        0 |           0 |      80 |
| Support Vector Machine | recall_at_least_0.90 |        0 |           1 |      30 |
| Support Vector Machine | recall_at_least_0.90 |        1 |           0 |       9 |
| Support Vector Machine | recall_at_least_0.90 |        1 |           1 |     122 |
| XGBoost                | max_f1               |        0 |           0 |      87 |
| XGBoost                | max_f1               |        0 |           1 |      23 |
| XGBoost                | max_f1               |        1 |           0 |      15 |
| XGBoost                | max_f1               |        1 |           1 |     116 |
| XGBoost                | recall_at_least_0.90 |        0 |           0 |      76 |
| XGBoost                | recall_at_least_0.90 |        0 |           1 |      34 |
| XGBoost                | recall_at_least_0.90 |        1 |           0 |       9 |
| XGBoost                | recall_at_least_0.90 |        1 |           1 |     122 |

## Exported Figures

- `outputs/figures/evaluation_oof_metric_comparison.png`
- `outputs/figures/evaluation_roc_curves.png`
- `outputs/figures/evaluation_precision_recall_curves.png`
- `outputs/figures/evaluation_threshold_sweep.png`
- `outputs/figures/evaluation_confusion_xgboost_default_0.50.png`
- `outputs/figures/evaluation_confusion_support_vector_machine_default_0.50.png`
- `outputs/figures/evaluation_confusion_support_vector_machine_recall_at_least_0.90.png`
- `outputs/figures/evaluation_confusion_xgboost_recall_at_least_0.90.png`

## Interpretation

- Best default-threshold OOF recall: XGBoost with recall 0.885.
- False negatives are the most clinically concerning error because they represent heart disease-positive patients predicted as negative.
- Threshold tuning demonstrates the trade-off between sensitivity and false alarms before final model selection.
- Final held-out test evaluation is deferred until Step 10 tuning is locked, preserving the test set as an unbiased final check.
