# Data Dictionary And Schema Validation

## Purpose

This document defines the authoritative schema for the heart disease prediction project. It separates true numerical features from encoded categorical variables, records human-readable labels for reporting, and validates observed categorical values before cleaning and modelling.

## Feature Groups

| feature_group       | columns                               |   n_columns |
|:--------------------|:--------------------------------------|------------:|
| numerical           | age, trestbps, chol, thalach, oldpeak |           5 |
| binary              | sex, fbs, exang, target               |           4 |
| nominal_categorical | cp, restecg, slope, thal              |           4 |
| ordinal_count       | ca                                    |           1 |

## Data Dictionary

| column   | description                                            | feature_group       | expected_dtype   | modeling_role   | valid_values                    | sentinel_missing_values   | labels                                                                                    |
|:---------|:-------------------------------------------------------|:--------------------|:-----------------|:----------------|:--------------------------------|:--------------------------|:------------------------------------------------------------------------------------------|
| age      | Patient age in years.                                  | numerical           | integer          | Predictor       | Positive adult age in years     |                           |                                                                                           |
| sex      | Patient sex encoded as a binary category.              | binary              | integer          | Predictor       | 0, 1                            |                           | 0 = Female; 1 = Male                                                                      |
| cp       | Chest pain type.                                       | nominal_categorical | integer          | Predictor       | 0, 1, 2, 3                      |                           | 0 = Typical angina; 1 = Atypical angina; 2 = Non-anginal pain; 3 = Asymptomatic           |
| trestbps | Resting blood pressure in mm Hg on admission.          | numerical           | integer          | Predictor       | Positive mm Hg value            |                           |                                                                                           |
| chol     | Serum cholesterol in mg/dl.                            | numerical           | integer          | Predictor       | Positive mg/dl value            |                           |                                                                                           |
| fbs      | Whether fasting blood sugar is greater than 120 mg/dl. | binary              | integer          | Predictor       | 0, 1                            |                           | 0 = Fasting blood sugar <= 120 mg/dl; 1 = Fasting blood sugar > 120 mg/dl                 |
| restecg  | Resting electrocardiographic result.                   | nominal_categorical | integer          | Predictor       | 0, 1, 2                         |                           | 0 = Normal; 1 = ST-T wave abnormality; 2 = Probable/definite left ventricular hypertrophy |
| thalach  | Maximum heart rate achieved.                           | numerical           | integer          | Predictor       | Positive beats-per-minute value |                           |                                                                                           |
| exang    | Whether exercise-induced angina is present.            | binary              | integer          | Predictor       | 0, 1                            |                           | 0 = No exercise-induced angina; 1 = Exercise-induced angina                               |
| oldpeak  | ST depression induced by exercise relative to rest.    | numerical           | float            | Predictor       | Non-negative continuous value   |                           |                                                                                           |
| slope    | Slope of the peak exercise ST segment.                 | nominal_categorical | integer          | Predictor       | 0, 1, 2                         |                           | 0 = Upsloping; 1 = Flat; 2 = Downsloping                                                  |
| ca       | Number of major vessels colored by fluoroscopy.        | ordinal_count       | integer          | Predictor       | 0, 1, 2, 3                      | 4                         | 0 = 0 major vessels; 1 = 1 major vessel; 2 = 2 major vessels; 3 = 3 major vessels         |
| thal     | Thalassemia category.                                  | nominal_categorical | integer          | Predictor       | 1, 2, 3                         | 0                         | 1 = Fixed defect; 2 = Normal; 3 = Reversible defect                                       |
| target   | Binary heart disease diagnosis label.                  | binary              | integer          | Target          | 0, 1                            |                           | 0 = No heart disease; 1 = Heart disease present                                           |

## Encoded-Value Validation

| column   | observed_values   | valid_values   | sentinel_missing_values   | observed_sentinel_values   | unexpected_values   |   n_sentinel_rows |   n_unexpected_rows | status                    |
|:---------|:------------------|:---------------|:--------------------------|:---------------------------|:--------------------|------------------:|--------------------:|:--------------------------|
| sex      | 0, 1              | 0, 1           |                           |                            |                     |                 0 |                   0 | pass                      |
| cp       | 0, 1, 2, 3        | 0, 1, 2, 3     |                           |                            |                     |                 0 |                   0 | pass                      |
| fbs      | 0, 1              | 0, 1           |                           |                            |                     |                 0 |                   0 | pass                      |
| restecg  | 0, 1, 2           | 0, 1, 2        |                           |                            |                     |                 0 |                   0 | pass                      |
| exang    | 0, 1              | 0, 1           |                           |                            |                     |                 0 |                   0 | pass                      |
| slope    | 0, 1, 2           | 0, 1, 2        |                           |                            |                     |                 0 |                   0 | pass                      |
| ca       | 0, 1, 2, 3, 4     | 0, 1, 2, 3     | 4                         | 4                          |                     |                 5 |                   0 | sentinel_missing_detected |
| thal     | 0, 1, 2, 3        | 1, 2, 3        | 0                         | 0                          |                     |                 2 |                   0 | sentinel_missing_detected |
| target   | 0, 1              | 0, 1           |                           |                            |                     |                 0 |                   0 | pass                      |

## Validation Summary

- No unexpected categorical encodings were found. Sentinel-coded missing values were detected and will be converted to `NaN` during Step 4 cleaning.
- Columns with sentinel-coded missing values: ca, thal
- Columns with unexpected invalid values: None

## Modelling Notes

- Numerical features will be imputed with the median and scaled in the modelling pipeline.
- Nominal categorical features will be imputed with the most frequent value and one-hot encoded in the modelling pipeline.
- `ca` is treated as an ordinal/count feature after sentinel decoding because values 0-3 represent ordered vessel counts.
- Binary variables are passed through after validation because their encodings are already machine-readable.
- `ca = 4` and `thal = 0` are not valid medical categories for this project schema; they are documented sentinel values and must be decoded to missing values in Step 4.
