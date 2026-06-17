# Baseline Model Comparison

## Scope

This report compares the baseline classifier lineup using stratified 5-fold cross-validation on the training split only. The held-out test set is not evaluated in this step.

## Split Summary

| split   |   rows |   target_0_count |   target_1_count |   target_1_percentage |
|:--------|-------:|-----------------:|-----------------:|----------------------:|
| train   |    241 |              110 |              131 |                 54.36 |
| test    |     61 |               28 |               33 |                 54.1  |

## Model Inventory

| model                  | estimator                      | notes                                                                                                         |
|:-----------------------|:-------------------------------|:--------------------------------------------------------------------------------------------------------------|
| Dummy Stratified       | DummyClassifier                | Chance-level baseline; every real model should beat it.                                                       |
| Logistic Regression    | LogisticRegression             | Linear baseline with balanced class weights.                                                                  |
| K-Nearest Neighbors    | KNeighborsClassifier           | Distance-based non-parametric classifier.                                                                     |
| Support Vector Machine | SVC                            | Margin-based classifier with probability estimates enabled.                                                   |
| Decision Tree          | DecisionTreeClassifier         | Interpretable tree baseline.                                                                                  |
| Random Forest          | RandomForestClassifier         | Bagged tree ensemble with balanced class weights.                                                             |
| Gaussian Naive Bayes   | GaussianNB                     | Included for lab coverage; GaussianNB assumes continuous features, which is imperfect after one-hot encoding. |
| Gradient Boosting      | GradientBoostingClassifier     | Scikit-learn boosted tree baseline.                                                                           |
| XGBoost                | SklearnCompatibleXGBClassifier | Industry-standard boosted tree classifier for tabular data.                                                   |

## Cross-Validation Summary

Models are sorted by recall mean, then ROC-AUC mean, then F1 mean. Recall is prioritized because false negatives are costly in a heart disease screening context.

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

## Per-Fold Scores

| model                  |   fold |   fit_time_seconds |   score_time_seconds |   accuracy |   precision |   recall |       f1 |   roc_auc |   average_precision |
|:-----------------------|-------:|-------------------:|---------------------:|-----------:|------------:|---------:|---------:|----------:|--------------------:|
| Dummy Stratified       |      1 |             0.0075 |               0.0125 |   0.530612 |    0.566667 | 0.62963  | 0.596491 |  0.51936  |            0.560872 |
| Dummy Stratified       |      2 |             0.0076 |               0.0117 |   0.5      |    0.533333 | 0.615385 | 0.571429 |  0.48951  |            0.536538 |
| Dummy Stratified       |      3 |             0.0078 |               0.0112 |   0.5      |    0.533333 | 0.615385 | 0.571429 |  0.48951  |            0.536538 |
| Dummy Stratified       |      4 |             0.0076 |               0.0122 |   0.5      |    0.533333 | 0.615385 | 0.571429 |  0.48951  |            0.536538 |
| Dummy Stratified       |      5 |             0.0081 |               0.0119 |   0.5      |    0.533333 | 0.615385 | 0.571429 |  0.48951  |            0.536538 |
| Logistic Regression    |      1 |             0.009  |               0.0113 |   0.877551 |    0.862069 | 0.925926 | 0.892857 |  0.949495 |            0.961977 |
| Logistic Regression    |      2 |             0.009  |               0.0116 |   0.708333 |    0.7      | 0.807692 | 0.75     |  0.854895 |            0.890409 |
| Logistic Regression    |      3 |             0.0096 |               0.0119 |   0.895833 |    0.92     | 0.884615 | 0.901961 |  0.93007  |            0.915485 |
| Logistic Regression    |      4 |             0.0091 |               0.012  |   0.875    |    0.884615 | 0.884615 | 0.884615 |  0.928322 |            0.940453 |
| Logistic Regression    |      5 |             0.0096 |               0.0114 |   0.854167 |    0.88     | 0.846154 | 0.862745 |  0.945804 |            0.963179 |
| K-Nearest Neighbors    |      1 |             0.0079 |               0.0517 |   0.816327 |    0.78125  | 0.925926 | 0.847458 |  0.872054 |            0.835446 |
| K-Nearest Neighbors    |      2 |             0.0076 |               0.0154 |   0.770833 |    0.758621 | 0.846154 | 0.8      |  0.828671 |            0.812905 |
| K-Nearest Neighbors    |      3 |             0.0074 |               0.0164 |   0.875    |    0.857143 | 0.923077 | 0.888889 |  0.909965 |            0.87486  |
| K-Nearest Neighbors    |      4 |             0.0079 |               0.0156 |   0.8125   |    0.869565 | 0.769231 | 0.816327 |  0.905594 |            0.879967 |
| K-Nearest Neighbors    |      5 |             0.0078 |               0.0153 |   0.833333 |    0.846154 | 0.846154 | 0.846154 |  0.91521  |            0.917402 |
| Support Vector Machine |      1 |             0.0109 |               0.0126 |   0.836735 |    0.806452 | 0.925926 | 0.862069 |  0.922559 |            0.936113 |
| Support Vector Machine |      2 |             0.0106 |               0.0128 |   0.770833 |    0.758621 | 0.846154 | 0.8      |  0.828671 |            0.826227 |
| Support Vector Machine |      3 |             0.0123 |               0.012  |   0.854167 |    0.851852 | 0.884615 | 0.867925 |  0.914336 |            0.892509 |
| Support Vector Machine |      4 |             0.0119 |               0.0119 |   0.875    |    0.884615 | 0.884615 | 0.884615 |  0.931818 |            0.939655 |
| Support Vector Machine |      5 |             0.0108 |               0.0116 |   0.833333 |    0.821429 | 0.884615 | 0.851852 |  0.945804 |            0.95895  |
| Decision Tree          |      1 |             0.0079 |               0.0111 |   0.755102 |    0.8      | 0.740741 | 0.769231 |  0.756734 |            0.73545  |
| Decision Tree          |      2 |             0.0084 |               0.0111 |   0.75     |    0.769231 | 0.769231 | 0.769231 |  0.748252 |            0.716716 |
| Decision Tree          |      3 |             0.008  |               0.0109 |   0.6875   |    0.689655 | 0.769231 | 0.727273 |  0.68007  |            0.655504 |
| Decision Tree          |      4 |             0.0082 |               0.0112 |   0.770833 |    0.758621 | 0.846154 | 0.8      |  0.763986 |            0.725243 |
| Decision Tree          |      5 |             0.0084 |               0.0119 |   0.854167 |    0.851852 | 0.884615 | 0.867925 |  0.851399 |            0.816061 |
| Random Forest          |      1 |             0.2048 |               0.0285 |   0.877551 |    0.862069 | 0.925926 | 0.892857 |  0.928451 |            0.941887 |
| Random Forest          |      2 |             0.2013 |               0.0264 |   0.729167 |    0.709677 | 0.846154 | 0.77193  |  0.829545 |            0.868946 |
| Random Forest          |      3 |             0.2102 |               0.0263 |   0.854167 |    0.851852 | 0.884615 | 0.867925 |  0.928322 |            0.92293  |
| Random Forest          |      4 |             0.2018 |               0.0264 |   0.833333 |    0.875    | 0.807692 | 0.84     |  0.928322 |            0.931623 |
| Random Forest          |      5 |             0.1996 |               0.0258 |   0.791667 |    0.807692 | 0.807692 | 0.807692 |  0.914336 |            0.929595 |
| Gaussian Naive Bayes   |      1 |             0.0079 |               0.0126 |   0.714286 |    0.685714 | 0.888889 | 0.774194 |  0.868687 |            0.930223 |
| Gaussian Naive Bayes   |      2 |             0.0079 |               0.0109 |   0.729167 |    0.724138 | 0.807692 | 0.763636 |  0.793706 |            0.840484 |
| Gaussian Naive Bayes   |      3 |             0.0074 |               0.0115 |   0.916667 |    0.892857 | 0.961538 | 0.925926 |  0.91958  |            0.890343 |
| Gaussian Naive Bayes   |      4 |             0.0074 |               0.0121 |   0.875    |    0.884615 | 0.884615 | 0.884615 |  0.905594 |            0.90945  |
| Gaussian Naive Bayes   |      5 |             0.0074 |               0.0108 |   0.833333 |    0.875    | 0.807692 | 0.84     |  0.907343 |            0.934879 |
| Gradient Boosting      |      1 |             0.0691 |               0.0117 |   0.816327 |    0.821429 | 0.851852 | 0.836364 |  0.90404  |            0.927871 |
| Gradient Boosting      |      2 |             0.0678 |               0.0123 |   0.708333 |    0.676471 | 0.884615 | 0.766667 |  0.823427 |            0.872752 |
| Gradient Boosting      |      3 |             0.0702 |               0.012  |   0.8125   |    0.84     | 0.807692 | 0.823529 |  0.902098 |            0.921622 |
| Gradient Boosting      |      4 |             0.0704 |               0.0119 |   0.8125   |    0.84     | 0.807692 | 0.823529 |  0.903846 |            0.919495 |
| Gradient Boosting      |      5 |             0.0706 |               0.0128 |   0.833333 |    0.821429 | 0.884615 | 0.851852 |  0.874126 |            0.888668 |
| XGBoost                |      1 |             0.26   |               0.0171 |   0.816327 |    0.8      | 0.888889 | 0.842105 |  0.914141 |            0.936047 |
| XGBoost                |      2 |             0.0284 |               0.0136 |   0.791667 |    0.766667 | 0.884615 | 0.821429 |  0.847902 |            0.887229 |
| XGBoost                |      3 |             0.028  |               0.0135 |   0.875    |    0.857143 | 0.923077 | 0.888889 |  0.917832 |            0.911146 |
| XGBoost                |      4 |             0.0281 |               0.014  |   0.854167 |    0.88     | 0.846154 | 0.862745 |  0.921329 |            0.927925 |
| XGBoost                |      5 |             0.028  |               0.0141 |   0.833333 |    0.821429 | 0.884615 | 0.851852 |  0.923077 |            0.937436 |

## Initial Observations

- Top model by recall: XGBoost with recall 0.885 +/- 0.027.
- Best non-dummy model by the project ranking: XGBoost.
- Dummy baseline recall: 0.618; real models must be interpreted relative to this baseline, not accuracy alone.
- Gaussian Naive Bayes is included for coverage, but its assumptions are imperfect after one-hot encoding.
- Hyperparameter tuning should focus on the strongest 2-3 models from this baseline ranking.
