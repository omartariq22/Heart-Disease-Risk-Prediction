# Hyperparameter Tuning Report

## Scope

This step tunes only the strongest baseline candidates: XGBoost, Support Vector Machine, and Logistic Regression. Grid search is performed on the training split only with stratified 5-fold cross-validation. The held-out test set is still not evaluated.

## Split Summary

| split   |   rows |   target_0_count |   target_1_count |   target_1_percentage |
|:--------|-------:|-----------------:|-----------------:|----------------------:|
| train   |    241 |              110 |              131 |                 54.36 |
| test    |     61 |               28 |               33 |                 54.1  |

## Search Space

| model                  |   n_parameter_combinations | parameter_grid                                                                                  |
|:-----------------------|---------------------------:|:------------------------------------------------------------------------------------------------|
| Logistic Regression    |                          4 | {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]}                |
| Support Vector Machine |                         12 | {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.1], "clf__kernel": ["linear", "rbf"]}        |
| XGBoost                |                          8 | {"clf__learning_rate": [0.03, 0.05], "clf__max_depth": [2, 3], "clf__n_estimators": [100, 200]} |

## Baseline Shortlist Metrics

| model                  |   accuracy_mean |   accuracy_std |   precision_mean |   precision_std |   recall_mean |   recall_std |   f1_mean |   f1_std |   roc_auc_mean |   roc_auc_std |   average_precision_mean |   average_precision_std |
|:-----------------------|----------------:|---------------:|-----------------:|----------------:|--------------:|-------------:|----------:|---------:|---------------:|--------------:|-------------------------:|------------------------:|
| XGBoost                |        0.834099 |       0.032375 |         0.825048 |        0.044996 |      0.88547  |     0.027263 |  0.853404 | 0.024995 |       0.904856 |      0.032022 |                 0.919957 |                0.021075 |
| Support Vector Machine |        0.834014 |       0.039005 |         0.824594 |        0.047532 |      0.885185 |     0.028214 |  0.853292 | 0.032072 |       0.908638 |      0.046205 |                 0.910691 |                0.053101 |
| Logistic Regression    |        0.842177 |       0.076265 |         0.849337 |        0.086084 |      0.8698   |     0.044737 |  0.858436 | 0.062333 |       0.921717 |      0.038504 |                 0.934301 |                0.031298 |

## Tuned Model Comparison

Models are sorted by recall mean, then ROC-AUC mean, then F1 mean. GridSearchCV refits on ROC-AUC because threshold tuning handles recall-specific operating points.

| model                  | best_params                                                                 |   best_refit_score |   accuracy_mean |   accuracy_std |   precision_mean |   precision_std |   recall_mean |   recall_std |   f1_mean |   f1_std |   roc_auc_mean |   roc_auc_std |   average_precision_mean |   average_precision_std |
|:-----------------------|:----------------------------------------------------------------------------|-------------------:|----------------:|---------------:|-----------------:|----------------:|--------------:|-------------:|----------:|---------:|---------------:|--------------:|-------------------------:|------------------------:|
| Support Vector Machine | {"clf__C": 0.1, "clf__gamma": "scale", "clf__kernel": "linear"}             |           0.921031 |        0.859014 |       0.054669 |         0.846551 |        0.058653 |      0.908262 |     0.039463 |  0.875794 | 0.046373 |       0.921031 |      0.037555 |                 0.928812 |                0.034115 |
| XGBoost                | {"clf__learning_rate": 0.05, "clf__max_depth": 2, "clf__n_estimators": 100} |           0.916175 |        0.85051  |       0.042803 |         0.843433 |        0.047318 |      0.892877 |     0.038164 |  0.866798 | 0.036614 |       0.916175 |      0.031525 |                 0.926632 |                0.023179 |
| Logistic Regression    | {"clf__C": 0.1, "clf__penalty": "l2", "clf__solver": "lbfgs"}               |           0.924501 |        0.858844 |       0.048419 |         0.859878 |        0.047782 |      0.885185 |     0.042665 |  0.872185 | 0.043697 |       0.924501 |      0.038629 |                 0.933305 |                0.036493 |

## Selected Tuned Candidate

| model                  | selection_reason                                           | best_params                                                     |   best_refit_score |   accuracy_mean |   accuracy_std |   precision_mean |   precision_std |   recall_mean |   recall_std |   f1_mean |   f1_std |   roc_auc_mean |   roc_auc_std |   average_precision_mean |   average_precision_std |
|:-----------------------|:-----------------------------------------------------------|:----------------------------------------------------------------|-------------------:|----------------:|---------------:|-----------------:|----------------:|--------------:|-------------:|----------:|---------:|---------------:|--------------:|-------------------------:|------------------------:|
| Support Vector Machine | Highest recall, then ROC-AUC and F1 among tuned candidates | {"clf__C": 0.1, "clf__gamma": "scale", "clf__kernel": "linear"} |           0.921031 |        0.859014 |       0.054669 |         0.846551 |        0.058653 |      0.908262 |     0.039463 |  0.875794 | 0.046373 |       0.921031 |      0.037555 |                 0.928812 |                0.034115 |

## Full Grid Results

| model                  |   rank_test_roc_auc | params                                                                      |   mean_test_accuracy |   std_test_accuracy |   mean_test_precision |   std_test_precision |   mean_test_recall |   std_test_recall |   mean_test_f1 |   std_test_f1 |   mean_test_roc_auc |   std_test_roc_auc |   mean_test_average_precision |   std_test_average_precision |
|:-----------------------|--------------------:|:----------------------------------------------------------------------------|---------------------:|--------------------:|----------------------:|---------------------:|-------------------:|------------------:|---------------:|--------------:|--------------------:|-------------------:|------------------------------:|-----------------------------:|
| Logistic Regression    |                   1 | {"clf__C": 0.1, "clf__penalty": "l2", "clf__solver": "lbfgs"}               |             0.858844 |           0.0484194 |              0.859878 |            0.0477822 |           0.885185 |         0.0426647 |       0.872185 |     0.0436972 |            0.924501 |          0.0386285 |                      0.933305 |                    0.0364932 |
| Logistic Regression    |                   2 | {"clf__C": 1, "clf__penalty": "l2", "clf__solver": "lbfgs"}                 |             0.842177 |           0.0682137 |              0.849337 |            0.0769957 |           0.869801 |         0.040014  |       0.858436 |     0.0557522 |            0.921717 |          0.0344394 |                      0.934301 |                    0.0279936 |
| Logistic Regression    |                   3 | {"clf__C": 10, "clf__penalty": "l2", "clf__solver": "lbfgs"}                |             0.83801  |           0.06799   |              0.842568 |            0.0754751 |           0.869801 |         0.040014  |       0.855117 |     0.0558905 |            0.915152 |          0.0358315 |                      0.929518 |                    0.0289152 |
| Logistic Regression    |                   4 | {"clf__C": 0.01, "clf__penalty": "l2", "clf__solver": "lbfgs"}              |             0.83818  |           0.0203187 |              0.834577 |            0.0302644 |           0.877778 |         0.0158984 |       0.855226 |     0.0162289 |            0.911733 |          0.0380791 |                      0.92297  |                    0.035568  |
| Support Vector Machine |                   1 | {"clf__C": 0.1, "clf__gamma": "scale", "clf__kernel": "linear"}             |             0.859014 |           0.0546687 |              0.846551 |            0.0586535 |           0.908262 |         0.0394625 |       0.875794 |     0.0463734 |            0.921031 |          0.0375555 |                      0.928812 |                    0.0341148 |
| Support Vector Machine |                   1 | {"clf__C": 0.1, "clf__gamma": 0.1, "clf__kernel": "linear"}                 |             0.859014 |           0.0546687 |              0.846551 |            0.0586535 |           0.908262 |         0.0394625 |       0.875794 |     0.0463734 |            0.921031 |          0.0375555 |                      0.928812 |                    0.0341148 |
| Support Vector Machine |                   3 | {"clf__C": 1, "clf__gamma": "scale", "clf__kernel": "linear"}               |             0.842262 |           0.0584729 |              0.837935 |            0.0663201 |           0.885185 |         0.0426647 |       0.860027 |     0.0492141 |            0.914103 |          0.038702  |                      0.925151 |                    0.0332195 |
| Support Vector Machine |                   3 | {"clf__C": 1, "clf__gamma": 0.1, "clf__kernel": "linear"}                   |             0.842262 |           0.0584729 |              0.837935 |            0.0663201 |           0.885185 |         0.0426647 |       0.860027 |     0.0492141 |            0.914103 |          0.038702  |                      0.925151 |                    0.0332195 |
| Support Vector Machine |                   5 | {"clf__C": 10, "clf__gamma": "scale", "clf__kernel": "linear"}              |             0.850595 |           0.0651344 |              0.839405 |            0.0676944 |           0.90057  |         0.0464519 |       0.868459 |     0.0555866 |            0.912393 |          0.0421601 |                      0.925712 |                    0.0374062 |
| Support Vector Machine |                   5 | {"clf__C": 10, "clf__gamma": 0.1, "clf__kernel": "linear"}                  |             0.850595 |           0.0651344 |              0.839405 |            0.0676944 |           0.90057  |         0.0464519 |       0.868459 |     0.0555866 |            0.912393 |          0.0421601 |                      0.925712 |                    0.0374062 |
| Support Vector Machine |                   7 | {"clf__C": 1, "clf__gamma": "scale", "clf__kernel": "rbf"}                  |             0.834014 |           0.0348874 |              0.824594 |            0.042514  |           0.885185 |         0.0252358 |       0.853292 |     0.0286859 |            0.908638 |          0.0413269 |                      0.910691 |                    0.0474949 |
| Support Vector Machine |                   8 | {"clf__C": 1, "clf__gamma": 0.1, "clf__kernel": "rbf"}                      |             0.834014 |           0.0348874 |              0.824594 |            0.042514  |           0.885185 |         0.0252358 |       0.853292 |     0.0286859 |            0.908625 |          0.0412584 |                      0.910562 |                    0.0474113 |
| Support Vector Machine |                   9 | {"clf__C": 0.1, "clf__gamma": "scale", "clf__kernel": "rbf"}                |             0.830017 |           0.0188622 |              0.80536  |            0.0232127 |           0.908547 |         0.0387548 |       0.853044 |     0.0162009 |            0.906592 |          0.0404267 |                      0.920479 |                    0.0406384 |
| Support Vector Machine |                   9 | {"clf__C": 0.1, "clf__gamma": 0.1, "clf__kernel": "rbf"}                    |             0.830017 |           0.0188622 |              0.80536  |            0.0232127 |           0.908547 |         0.0387548 |       0.853044 |     0.0162009 |            0.906592 |          0.0404267 |                      0.920479 |                    0.0406384 |
| Support Vector Machine |                  11 | {"clf__C": 10, "clf__gamma": 0.1, "clf__kernel": "rbf"}                     |             0.796684 |           0.0203094 |              0.810504 |            0.0426762 |           0.824217 |         0.0527227 |       0.814792 |     0.0182934 |            0.879202 |          0.0286664 |                      0.88855  |                    0.0397779 |
| Support Vector Machine |                  12 | {"clf__C": 10, "clf__gamma": "scale", "clf__kernel": "rbf"}                 |             0.792602 |           0.0220309 |              0.809026 |            0.0433754 |           0.816809 |         0.0508896 |       0.810506 |     0.0187411 |            0.878166 |          0.0280392 |                      0.887141 |                    0.0388626 |
| XGBoost                |                   1 | {"clf__learning_rate": 0.05, "clf__max_depth": 2, "clf__n_estimators": 100} |             0.85051  |           0.0428031 |              0.843433 |            0.0473179 |           0.892877 |         0.0381639 |       0.866798 |     0.036614  |            0.916175 |          0.0315251 |                      0.926632 |                    0.0231792 |
| XGBoost                |                   2 | {"clf__learning_rate": 0.03, "clf__max_depth": 3, "clf__n_estimators": 100} |             0.846429 |           0.0340172 |              0.837461 |            0.040178  |           0.893162 |         0.0375485 |       0.863569 |     0.0285303 |            0.913442 |          0.0321951 |                      0.925953 |                    0.0226947 |
| XGBoost                |                   3 | {"clf__learning_rate": 0.03, "clf__max_depth": 2, "clf__n_estimators": 200} |             0.846344 |           0.0390366 |              0.837084 |            0.0426968 |           0.892877 |         0.0381639 |       0.863444 |     0.0335392 |            0.913054 |          0.0314759 |                      0.923667 |                    0.0227747 |
| XGBoost                |                   4 | {"clf__learning_rate": 0.03, "clf__max_depth": 3, "clf__n_estimators": 200} |             0.838095 |           0.0280205 |              0.824945 |            0.0315138 |           0.892877 |         0.0294069 |       0.857103 |     0.0238079 |            0.912743 |          0.0342838 |                      0.927587 |                    0.0232755 |
| XGBoost                |                   5 | {"clf__learning_rate": 0.05, "clf__max_depth": 3, "clf__n_estimators": 100} |             0.842262 |           0.0285466 |              0.831714 |            0.0382563 |           0.892877 |         0.0294069 |       0.860421 |     0.023199  |            0.911629 |          0.0316465 |                      0.923126 |                    0.0230489 |
| XGBoost                |                   6 | {"clf__learning_rate": 0.05, "clf__max_depth": 2, "clf__n_estimators": 200} |             0.838095 |           0.0309638 |              0.826049 |            0.0413335 |           0.892877 |         0.0294069 |       0.857324 |     0.0250931 |            0.911435 |          0.0232111 |                      0.922719 |                    0.0167523 |
| XGBoost                |                   7 | {"clf__learning_rate": 0.03, "clf__max_depth": 2, "clf__n_estimators": 100} |             0.825595 |           0.0743118 |              0.816931 |            0.0787347 |           0.88547  |         0.0344435 |       0.848646 |     0.0572288 |            0.910256 |          0.0378384 |                      0.918249 |                    0.0300908 |
| XGBoost                |                   8 | {"clf__learning_rate": 0.05, "clf__max_depth": 3, "clf__n_estimators": 200} |             0.834099 |           0.028957  |              0.825048 |            0.040246  |           0.88547  |         0.0243852 |       0.853404 |     0.0223559 |            0.904856 |          0.0286416 |                      0.919957 |                    0.0188502 |

## Exported Figures

- `outputs/figures/tuning_model_comparison.png`

## Decision

- Selected tuned candidate before final thresholding: Support Vector Machine.
- Best parameters: `{"clf__C": 0.1, "clf__gamma": "scale", "clf__kernel": "linear"}`.
- Tuned recall: 0.908; tuned ROC-AUC: 0.921; tuned F1: 0.876.
- The held-out test set remains untouched; final test evaluation should happen only after threshold choice is locked for the selected tuned model.
