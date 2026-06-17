# Feature Importance And Medical Insight Report

## Scope

This interpretation step fits tuned candidate pipelines on the training split only. The held-out test set remains untouched. Interpretability results are educational and exploratory; they do not establish clinical causality.

## Split Summary

| split   |   rows |   target_0_count |   target_1_count |   target_1_percentage |
|:--------|-------:|-----------------:|-----------------:|----------------------:|
| train   |    241 |              110 |              131 |                 54.36 |
| test    |     61 |               28 |               33 |                 54.1  |

## Selected Model For Interpretation

| selected_model         | best_params                                                     | interpretation_basis                                       |
|:-----------------------|:----------------------------------------------------------------|:-----------------------------------------------------------|
| Support Vector Machine | {"clf__C": 0.1, "clf__gamma": "scale", "clf__kernel": "linear"} | Tuned linear SVM coefficients plus permutation importance. |

## Cross-Method Insight Summary

| method                                 | top_features                                                       | interpretation                                                              |
|:---------------------------------------|:-------------------------------------------------------------------|:----------------------------------------------------------------------------|
| Tuned SVM coefficients                 | cp_0.0, thal_2.0, ca, thal_3.0, exang, sex, oldpeak, cp_2.0        | Largest absolute linear effects in the selected tuned model.                |
| Tuned Logistic Regression coefficients | cp_0.0, thal_2.0, ca, thal_3.0, sex, exang, oldpeak, cp_2.0        | Linear coefficient sanity check from a strong calibrated baseline.          |
| Tuned XGBoost importances              | thal_2.0, cp_0.0, ca, exang, thalach, oldpeak, thal_3.0, slope_2.0 | Tree-based split/gain importance from the strongest boosted-tree candidate. |
| Permutation importance                 | thal, cp, ca, oldpeak, chol, sex, exang, slope                     | Model-agnostic ROC-AUC drop after shuffling raw input features.             |

## Tuned SVM Coefficients

Positive coefficients increase the model score for heart disease presence; negative coefficients decrease it.

| model                  | transformed_feature   | feature     |   coefficient |   abs_coefficient | direction                      |
|:-----------------------|:----------------------|:------------|--------------:|------------------:|:-------------------------------|
| Support Vector Machine | nominal__cp_0.0       | cp_0.0      |    -0.65707   |         0.65707   | decreases positive-class score |
| Support Vector Machine | nominal__thal_2.0     | thal_2.0    |     0.560527  |         0.560527  | increases positive-class score |
| Support Vector Machine | ordinal__ca           | ca          |    -0.497654  |         0.497654  | decreases positive-class score |
| Support Vector Machine | nominal__thal_3.0     | thal_3.0    |    -0.460527  |         0.460527  | decreases positive-class score |
| Support Vector Machine | binary__exang         | exang       |    -0.406425  |         0.406425  | decreases positive-class score |
| Support Vector Machine | binary__sex           | sex         |    -0.362294  |         0.362294  | decreases positive-class score |
| Support Vector Machine | numeric__oldpeak      | oldpeak     |    -0.327229  |         0.327229  | decreases positive-class score |
| Support Vector Machine | nominal__cp_2.0       | cp_2.0      |     0.306194  |         0.306194  | increases positive-class score |
| Support Vector Machine | nominal__cp_3.0       | cp_3.0      |     0.294597  |         0.294597  | increases positive-class score |
| Support Vector Machine | nominal__slope_1.0    | slope_1.0   |    -0.19501   |         0.19501   | decreases positive-class score |
| Support Vector Machine | numeric__chol         | chol        |    -0.188906  |         0.188906  | decreases positive-class score |
| Support Vector Machine | nominal__slope_2.0    | slope_2.0   |     0.17227   |         0.17227   | increases positive-class score |
| Support Vector Machine | numeric__thalach      | thalach     |     0.127072  |         0.127072  | increases positive-class score |
| Support Vector Machine | nominal__restecg_1.0  | restecg_1.0 |     0.103036  |         0.103036  | increases positive-class score |
| Support Vector Machine | nominal__thal_1.0     | thal_1.0    |    -0.1       |         0.1       | decreases positive-class score |
| Support Vector Machine | numeric__age          | age         |     0.0633967 |         0.0633967 | increases positive-class score |
| Support Vector Machine | nominal__restecg_0.0  | restecg_0.0 |    -0.0627324 |         0.0627324 | decreases positive-class score |
| Support Vector Machine | nominal__cp_1.0       | cp_1.0      |     0.0562788 |         0.0562788 | increases positive-class score |
| Support Vector Machine | binary__fbs           | fbs         |     0.0531002 |         0.0531002 | increases positive-class score |
| Support Vector Machine | nominal__restecg_2.0  | restecg_2.0 |    -0.0403039 |         0.0403039 | decreases positive-class score |
| Support Vector Machine | numeric__trestbps     | trestbps    |    -0.0274001 |         0.0274001 | decreases positive-class score |
| Support Vector Machine | nominal__slope_0.0    | slope_0.0   |     0.0227402 |         0.0227402 | increases positive-class score |

## Tuned Logistic Regression Coefficients

| model               | transformed_feature   | feature     |   coefficient |   abs_coefficient | direction                      |
|:--------------------|:----------------------|:------------|--------------:|------------------:|:-------------------------------|
| Logistic Regression | nominal__cp_0.0       | cp_0.0      |   -0.660573   |        0.660573   | decreases positive-class score |
| Logistic Regression | nominal__thal_2.0     | thal_2.0    |    0.594719   |        0.594719   | increases positive-class score |
| Logistic Regression | ordinal__ca           | ca          |   -0.553636   |        0.553636   | decreases positive-class score |
| Logistic Regression | nominal__thal_3.0     | thal_3.0    |   -0.519367   |        0.519367   | decreases positive-class score |
| Logistic Regression | binary__sex           | sex         |   -0.437883   |        0.437883   | decreases positive-class score |
| Logistic Regression | binary__exang         | exang       |   -0.434824   |        0.434824   | decreases positive-class score |
| Logistic Regression | numeric__oldpeak      | oldpeak     |   -0.391355   |        0.391355   | decreases positive-class score |
| Logistic Regression | nominal__cp_2.0       | cp_2.0      |    0.383878   |        0.383878   | increases positive-class score |
| Logistic Regression | numeric__thalach      | thalach     |    0.319393   |        0.319393   | increases positive-class score |
| Logistic Regression | numeric__chol         | chol        |   -0.261911   |        0.261911   | decreases positive-class score |
| Logistic Regression | nominal__slope_1.0    | slope_1.0   |   -0.222171   |        0.222171   | decreases positive-class score |
| Logistic Regression | nominal__slope_2.0    | slope_2.0   |    0.213901   |        0.213901   | increases positive-class score |
| Logistic Regression | nominal__cp_3.0       | cp_3.0      |    0.176074   |        0.176074   | increases positive-class score |
| Logistic Regression | nominal__restecg_1.0  | restecg_1.0 |    0.162617   |        0.162617   | increases positive-class score |
| Logistic Regression | nominal__restecg_0.0  | restecg_0.0 |   -0.147878   |        0.147878   | decreases positive-class score |
| Logistic Regression | numeric__trestbps     | trestbps    |   -0.139106   |        0.139106   | decreases positive-class score |
| Logistic Regression | nominal__cp_1.0       | cp_1.0      |    0.101239   |        0.101239   | increases positive-class score |
| Logistic Regression | nominal__thal_1.0     | thal_1.0    |   -0.0747348  |        0.0747348  | decreases positive-class score |
| Logistic Regression | numeric__age          | age         |    0.0286933  |        0.0286933  | increases positive-class score |
| Logistic Regression | nominal__restecg_2.0  | restecg_2.0 |   -0.0141223  |        0.0141223  | decreases positive-class score |
| Logistic Regression | nominal__slope_0.0    | slope_0.0   |    0.00888706 |        0.00888706 | increases positive-class score |
| Logistic Regression | binary__fbs           | fbs         |    0.00701976 |        0.00701976 | increases positive-class score |

## Tuned XGBoost Feature Importances

| model   | transformed_feature   | feature     |   importance |
|:--------|:----------------------|:------------|-------------:|
| XGBoost | nominal__thal_2.0     | thal_2.0    |   0.218009   |
| XGBoost | nominal__cp_0.0       | cp_0.0      |   0.151107   |
| XGBoost | ordinal__ca           | ca          |   0.0718103  |
| XGBoost | binary__exang         | exang       |   0.0634507  |
| XGBoost | numeric__thalach      | thalach     |   0.058965   |
| XGBoost | numeric__oldpeak      | oldpeak     |   0.0523455  |
| XGBoost | nominal__thal_3.0     | thal_3.0    |   0.0475372  |
| XGBoost | nominal__slope_2.0    | slope_2.0   |   0.0440356  |
| XGBoost | numeric__age          | age         |   0.0376785  |
| XGBoost | nominal__restecg_1.0  | restecg_1.0 |   0.0359153  |
| XGBoost | numeric__chol         | chol        |   0.035849   |
| XGBoost | binary__sex           | sex         |   0.0340135  |
| XGBoost | nominal__cp_3.0       | cp_3.0      |   0.033883   |
| XGBoost | nominal__cp_2.0       | cp_2.0      |   0.033496   |
| XGBoost | nominal__restecg_0.0  | restecg_0.0 |   0.0276952  |
| XGBoost | nominal__slope_1.0    | slope_1.0   |   0.0266403  |
| XGBoost | numeric__trestbps     | trestbps    |   0.0209722  |
| XGBoost | nominal__slope_0.0    | slope_0.0   |   0.00659632 |
| XGBoost | nominal__restecg_2.0  | restecg_2.0 |   0          |
| XGBoost | nominal__cp_1.0       | cp_1.0      |   0          |
| XGBoost | nominal__thal_1.0     | thal_1.0    |   0          |
| XGBoost | binary__fbs           | fbs         |   0          |

## Tuned SVM Permutation Importance

| model                  | feature   |   importance_mean |   importance_std |
|:-----------------------|:----------|------------------:|-----------------:|
| Support Vector Machine | thal      |       0.0436086   |      0.0129685   |
| Support Vector Machine | cp        |       0.0386954   |      0.00778657  |
| Support Vector Machine | ca        |       0.0310618   |      0.00967065  |
| Support Vector Machine | oldpeak   |       0.0128383   |      0.00469295  |
| Support Vector Machine | chol      |       0.00922276  |      0.00405668  |
| Support Vector Machine | sex       |       0.00693616  |      0.00310896  |
| Support Vector Machine | exang     |       0.00692575  |      0.00340095  |
| Support Vector Machine | slope     |       0.00480569  |      0.00279968  |
| Support Vector Machine | thalach   |       0.00355309  |      0.00203529  |
| Support Vector Machine | restecg   |       0.00288341  |      0.00198799  |
| Support Vector Machine | trestbps  |       0.000496183 |      0.000529542 |
| Support Vector Machine | fbs       |       5.89868e-05 |      0.000339457 |
| Support Vector Machine | age       |      -0.000117974 |      0.000805445 |

## Exported Figures

- `outputs/figures/interpret_svm_coefficients.png`
- `outputs/figures/interpret_logistic_coefficients.png`
- `outputs/figures/interpret_xgboost_importances.png`
- `outputs/figures/interpret_permutation_importance.png`

## Project Question Connections

- Chest pain type, thalassemia category, exercise-induced angina, vessel count, `oldpeak`, and maximum heart rate repeatedly appear as important signals across model families.
- Sex appears in the linear models and EDA as a meaningful association in this dataset, but this should be framed as dataset-specific rather than a universal medical conclusion.
- Age has weaker model importance than several exercise/ECG-related variables, even though age-group EDA remains useful for answering the project's demographic questions.
- The strongest model signals align with clinically plausible cardiac stress and diagnostic markers, but the dataset is small and historical, so conclusions remain educational.
