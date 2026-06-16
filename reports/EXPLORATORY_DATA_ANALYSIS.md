# Exploratory Data Analysis

## Scope

This EDA is performed on the cleaned training split only. The held-out test set is not used for visual exploration, statistical testing, feature selection, or decision-making.

## Split Summary

| split   |   rows |   target_0_count |   target_1_count |   target_1_percentage |
|:--------|-------:|-----------------:|-----------------:|----------------------:|
| train   |    241 |              110 |              131 |                 54.36 |
| test    |     61 |               28 |               33 |                 54.1  |

## Target Distribution

|   target |   count | label                 |   percentage |
|---------:|--------:|:----------------------|-------------:|
|        0 |     110 | No heart disease      |        45.64 |
|        1 |     131 | Heart disease present |        54.36 |

## Age-Based Analysis

| age_group   |   rows |   heart_disease_cases |   heart_disease_percentage |
|:------------|-------:|----------------------:|---------------------------:|
| <40         |     12 |                     9 |                      75    |
| 40-49       |     59 |                    40 |                      67.8  |
| 50-59       |     98 |                    51 |                      52.04 |
| 60-69       |     63 |                    25 |                      39.68 |
| 70+         |      9 |                     6 |                      66.67 |

Younger patients (`<40`) have a heart disease-positive rate of 75.00% in the training split.

## Sex-Based Analysis

|   sex | sex_label   |   rows |   heart_disease_cases |   heart_disease_percentage |
|------:|:------------|-------:|----------------------:|---------------------------:|
|     0 | Female      |     76 |                    55 |                      72.37 |
|     1 | Male        |    165 |                    76 |                      46.06 |

Female positive rate: 72.37%. Male positive rate: 46.06%. In this dataset split, female patients show the higher positive-class percentage, while male patients are more numerous overall.

## Numerical Summary By Target

|   target | target_label          | feature   |    mean |   median |    std |   min |   max |
|---------:|:----------------------|:----------|--------:|---------:|-------:|------:|------:|
|        0 | No heart disease      | age       |  56.455 |    58    |  8.004 |    35 |  77   |
|        0 | No heart disease      | trestbps  | 134.409 |   130    | 19.479 |   100 | 200   |
|        0 | No heart disease      | chol      | 252.136 |   249    | 50.822 |   131 | 409   |
|        0 | No heart disease      | thalach   | 140     |   142    | 21.633 |    88 | 195   |
|        0 | No heart disease      | oldpeak   |   1.659 |     1.45 |  1.336 |     0 |   6.2 |
|        1 | Heart disease present | age       |  52.565 |    52    |  9.812 |    29 |  76   |
|        1 | Heart disease present | trestbps  | 128.962 |   130    | 16.297 |    94 | 180   |
|        1 | Heart disease present | chol      | 237.718 |   233    | 44.845 |   126 | 394   |
|        1 | Heart disease present | thalach   | 158.527 |   161    | 18.718 |   105 | 202   |
|        1 | Heart disease present | oldpeak   |   0.577 |     0.2  |  0.807 |     0 |   4.2 |

## Categorical Target Rates

| feature   |   value | label                                          |   rows |   heart_disease_cases |   heart_disease_percentage |
|:----------|--------:|:-----------------------------------------------|-------:|----------------------:|---------------------------:|
| sex       |       0 | Female                                         |     76 |                    55 |                      72.37 |
| sex       |       1 | Male                                           |    165 |                    76 |                      46.06 |
| cp        |       0 | Typical angina                                 |    114 |                    29 |                      25.44 |
| cp        |       1 | Atypical angina                                |     43 |                    35 |                      81.4  |
| cp        |       2 | Non-anginal pain                               |     69 |                    56 |                      81.16 |
| cp        |       3 | Asymptomatic                                   |     15 |                    11 |                      73.33 |
| fbs       |       0 | Fasting blood sugar <= 120 mg/dl               |    207 |                   114 |                      55.07 |
| fbs       |       1 | Fasting blood sugar > 120 mg/dl                |     34 |                    17 |                      50    |
| restecg   |       0 | Normal                                         |    112 |                    50 |                      44.64 |
| restecg   |       1 | ST-T wave abnormality                          |    125 |                    80 |                      64    |
| restecg   |       2 | Probable/definite left ventricular hypertrophy |      4 |                     1 |                      25    |
| exang     |       0 | No exercise-induced angina                     |    159 |                   112 |                      70.44 |
| exang     |       1 | Exercise-induced angina                        |     82 |                    19 |                      23.17 |
| slope     |       0 | Upsloping                                      |     18 |                     7 |                      38.89 |
| slope     |       1 | Flat                                           |    109 |                    37 |                      33.94 |
| slope     |       2 | Downsloping                                    |    114 |                    87 |                      76.32 |
| ca        |       0 | 0 major vessels                                |    138 |                   101 |                      73.19 |
| ca        |       1 | 1 major vessel                                 |     55 |                    19 |                      34.55 |
| ca        |       2 | 2 major vessels                                |     27 |                     5 |                      18.52 |
| ca        |       3 | 3 major vessels                                |     17 |                     3 |                      17.65 |
| thal      |       1 | Fixed defect                                   |     13 |                     3 |                      23.08 |
| thal      |       2 | Normal                                         |    129 |                   104 |                      80.62 |
| thal      |       3 | Reversible defect                              |     98 |                    23 |                      23.47 |

## Pearson Correlation

|          |    age |   trestbps |   chol |   thalach |   oldpeak |   target |
|:---------|-------:|-----------:|-------:|----------:|----------:|---------:|
| age      |  1     |      0.303 |  0.22  |    -0.406 |     0.176 |   -0.211 |
| trestbps |  0.303 |      1     |  0.198 |    -0.059 |     0.196 |   -0.151 |
| chol     |  0.22  |      0.198 |  1     |    -0.06  |     0.057 |   -0.15  |
| thalach  | -0.406 |     -0.059 | -0.06  |     1     |    -0.369 |    0.419 |
| oldpeak  |  0.176 |      0.196 |  0.057 |    -0.369 |     1     |   -0.448 |
| target   | -0.211 |     -0.151 | -0.15  |     0.419 |    -0.448 |    1     |

## Spearman Correlation

|          |    age |   trestbps |   chol |   thalach |   oldpeak |   target |
|:---------|-------:|-----------:|-------:|----------:|----------:|---------:|
| age      |  1     |      0.312 |  0.219 |    -0.404 |     0.23  |   -0.223 |
| trestbps |  0.312 |      1     |  0.184 |    -0.058 |     0.132 |   -0.126 |
| chol     |  0.219 |      0.184 |  1     |    -0.073 |     0.039 |   -0.157 |
| thalach  | -0.404 |     -0.058 | -0.073 |     1     |    -0.455 |    0.426 |
| oldpeak  |  0.23  |      0.132 |  0.039 |    -0.455 |     1     |   -0.442 |
| target   | -0.223 |     -0.126 | -0.157 |     0.426 |    -0.442 |    1     |

## Chi-Square Tests For Categorical Predictors

| feature   |    chi2 |   p_value |   degrees_of_freedom | significant_at_0_05   |
|:----------|--------:|----------:|---------------------:|:----------------------|
| cp        | 73.2523 |  0        |                    3 | True                  |
| ca        | 51.5653 |  0        |                    3 | True                  |
| slope     | 42.1969 |  0        |                    2 | True                  |
| exang     | 46.8355 |  0        |                    1 | True                  |
| thal      | 78.6202 |  0        |                    2 | True                  |
| sex       | 13.4741 |  0.000242 |                    1 | True                  |
| restecg   | 10.3343 |  0.005701 |                    2 | True                  |
| fbs       |  0.1329 |  0.715431 |                    1 | False                 |

Significant categorical predictors at alpha = 0.05: cp, ca, slope, exang, thal, sex, restecg.

## Mann-Whitney U Tests For Numerical Predictors

| feature   |   mann_whitney_u |   p_value | significant_at_0_05   |
|:----------|-----------------:|----------:|:----------------------|
| thalach   |           3648.5 |  0        | True                  |
| oldpeak   |          10827.5 |  0        | True                  |
| age       |           9066   |  0.000553 | True                  |
| chol      |           8520   |  0.014746 | True                  |
| trestbps  |           8251   |  0.051817 | False                 |

Significant numerical predictors at alpha = 0.05: thalach, oldpeak, age, chol.

## Mutual Information

| feature   |   mutual_information |
|:----------|---------------------:|
| thal      |             0.174561 |
| cp        |             0.160729 |
| ca        |             0.113436 |
| exang     |             0.104632 |
| slope     |             0.09073  |
| oldpeak   |             0.084925 |
| chol      |             0.0746   |
| thalach   |             0.047741 |
| trestbps  |             0.032027 |
| sex       |             0.03103  |
| restecg   |             0.021649 |
| fbs       |             0.000626 |
| age       |             0        |

Top mutual-information features:

| feature   |   mutual_information |
|:----------|---------------------:|
| thal      |             0.174561 |
| cp        |             0.160729 |
| ca        |             0.113436 |
| exang     |             0.104632 |
| slope     |             0.09073  |

## Exported Figures

- `outputs/figures/eda_target_distribution.png`
- `outputs/figures/eda_age_distribution_by_target.png`
- `outputs/figures/eda_age_group_target_rate.png`
- `outputs/figures/eda_sex_target_rate.png`
- `outputs/figures/eda_pearson_correlation_heatmap.png`
- `outputs/figures/eda_chest_pain_vs_target.png`
- `outputs/figures/eda_exang_vs_target.png`
- `outputs/figures/eda_thalach_by_target.png`
- `outputs/figures/eda_cholesterol_by_target.png`
- `outputs/figures/eda_mutual_information.png`

## Initial EDA Observations

- The training split preserves the mild class imbalance seen in the cleaned dataset.
- Age-group and sex-based target-rate tables directly answer the required project questions without using the test set.
- `thalach`, `oldpeak`, and several encoded clinical variables show meaningful target association and should be watched during modelling and interpretation.
- Statistical tests and mutual-information scores are exploratory evidence only; final feature importance must come from validated models in later steps.
