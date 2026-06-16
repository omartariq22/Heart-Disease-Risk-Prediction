# Preprocessing Pipeline Report

## Scope

This report documents the shared preprocessing `ColumnTransformer` that every classification model will use. The transformer is fitted on the training split only and then applied to validation/test data through scikit-learn `Pipeline` objects.

## Split Summary

| split   |   rows |   target_0_count |   target_1_count |   target_1_percentage |
|:--------|-------:|-----------------:|-----------------:|----------------------:|
| train   |    241 |              110 |              131 |                 54.36 |
| test    |     61 |               28 |               33 |                 54.1  |

## Feature And Target Shapes

| object | shape |
|:--|:--|
| `X_train` | (241, 13) |
| `X_test` | (61, 13) |
| `y_train` | (241,) |
| `y_test` | (61,) |
| transformed training matrix | (241, 22) |
| transformed test matrix | (61, 22) |

## Column Routing

| feature   | group   | pipeline_steps                                                                                      |
|:----------|:--------|:----------------------------------------------------------------------------------------------------|
| age       | numeric | SimpleImputer(strategy='median') -> StandardScaler()                                                |
| trestbps  | numeric | SimpleImputer(strategy='median') -> StandardScaler()                                                |
| chol      | numeric | SimpleImputer(strategy='median') -> StandardScaler()                                                |
| thalach   | numeric | SimpleImputer(strategy='median') -> StandardScaler()                                                |
| oldpeak   | numeric | SimpleImputer(strategy='median') -> StandardScaler()                                                |
| cp        | nominal | SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', drop='if_binary') |
| restecg   | nominal | SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', drop='if_binary') |
| slope     | nominal | SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', drop='if_binary') |
| thal      | nominal | SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', drop='if_binary') |
| ca        | ordinal | SimpleImputer(strategy='most_frequent')                                                             |
| sex       | binary  | passthrough                                                                                         |
| fbs       | binary  | passthrough                                                                                         |
| exang     | binary  | passthrough                                                                                         |

## Missing Values Before Pipeline Imputation

| feature   |   train_missing_count |   test_missing_count |
|:----------|----------------------:|---------------------:|
| age       |                     0 |                    0 |
| sex       |                     0 |                    0 |
| cp        |                     0 |                    0 |
| trestbps  |                     0 |                    0 |
| chol      |                     0 |                    0 |
| fbs       |                     0 |                    0 |
| restecg   |                     0 |                    0 |
| thalach   |                     0 |                    0 |
| exang     |                     0 |                    0 |
| oldpeak   |                     0 |                    0 |
| slope     |                     0 |                    0 |
| ca        |                     4 |                    0 |
| thal      |                     1 |                    1 |

## Transformed Feature Names

| transformed_feature   |
|:----------------------|
| numeric__age          |
| numeric__trestbps     |
| numeric__chol         |
| numeric__thalach      |
| numeric__oldpeak      |
| nominal__cp_0.0       |
| nominal__cp_1.0       |
| nominal__cp_2.0       |
| nominal__cp_3.0       |
| nominal__restecg_0.0  |
| nominal__restecg_1.0  |
| nominal__restecg_2.0  |
| nominal__slope_0.0    |
| nominal__slope_1.0    |
| nominal__slope_2.0    |
| nominal__thal_1.0     |
| nominal__thal_2.0     |
| nominal__thal_3.0     |
| ordinal__ca           |
| binary__sex           |
| binary__fbs           |
| binary__exang         |

## Leakage-Control Notes

- Deterministic cleaning happens before splitting because duplicate removal and sentinel decoding do not estimate parameters from the data.
- Median, most-frequent imputation, scaling, and one-hot encoding are fitted inside the `ColumnTransformer`.
- Later model training must wrap this preprocessor inside `Pipeline([('prep', preprocessor), ('clf', estimator)])`.
- The held-out test set is transformed only by a preprocessor fitted on the training data.
- Age groups remain EDA-only and are not included as model features.
