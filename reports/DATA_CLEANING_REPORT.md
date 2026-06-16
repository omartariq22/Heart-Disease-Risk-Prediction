# Data Cleaning Report

## Purpose

This report documents the deterministic data-cleaning actions applied after the raw-data audit and schema validation. The cleaned snapshot is generated for inspection and reporting; modelling code should still read from `data/raw/heart.csv`, call the same cleaning function, and perform imputation inside the modelling pipeline to avoid leakage.

## Cleaning Actions

- Dropped exact duplicate rows before any train/test split.
- Decoded documented sentinel missing values to `NaN`:
  - `ca = 4`
  - `thal = 0`
- Preserved medically plausible clinical extremes for later outlier analysis.
- Did not impute missing values in the dataframe; imputation belongs inside the modelling pipeline.

## Cleaning Summary

| metric                         |   value |
|:-------------------------------|--------:|
| raw_rows                       |     303 |
| raw_columns                    |      14 |
| duplicate_rows_removed         |       1 |
| cleaned_rows                   |     302 |
| cleaned_columns                |      14 |
| raw_explicit_missing_values    |       0 |
| cleaned_missing_values         |       6 |
| sentinel_values_decoded_to_nan |       6 |
| target_0_count_after_cleaning  |     138 |
| target_1_count_after_cleaning  |     164 |

## Duplicate Records

|   original_index |   age |   sex |   cp |   trestbps |   chol |   fbs |   restecg |   thalach |   exang |   oldpeak |   slope |   ca |   thal |   target |
|-----------------:|------:|------:|-----:|-----------:|-------:|------:|----------:|----------:|--------:|----------:|--------:|-----:|-------:|---------:|
|              163 |    38 |     1 |    2 |        138 |    175 |     0 |         1 |       173 |       0 |         0 |       2 |    4 |      2 |        1 |
|              164 |    38 |     1 |    2 |        138 |    175 |     0 |         1 |       173 |       0 |         0 |       2 |    4 |      2 |        1 |

## Sentinel Replacements

| column   |   sentinel_values |   n_replaced_with_nan |
|:---------|------------------:|----------------------:|
| ca       |                 4 |                     4 |
| thal     |                 0 |                     2 |

## Missing Values After Cleaning

| column   |   raw_missing_count |   cleaned_missing_count |
|:---------|--------------------:|------------------------:|
| age      |                   0 |                       0 |
| sex      |                   0 |                       0 |
| cp       |                   0 |                       0 |
| trestbps |                   0 |                       0 |
| chol     |                   0 |                       0 |
| fbs      |                   0 |                       0 |
| restecg  |                   0 |                       0 |
| thalach  |                   0 |                       0 |
| exang    |                   0 |                       0 |
| oldpeak  |                   0 |                       0 |
| slope    |                   0 |                       0 |
| ca       |                   0 |                       4 |
| thal     |                   0 |                       2 |
| target   |                   0 |                       0 |

## Numeric Range Validation

| column   |   observed_min |   observed_max |   allowed_min |   allowed_max | unit             |   n_below_min |   n_above_max | status   |
|:---------|---------------:|---------------:|--------------:|--------------:|:-----------------|--------------:|--------------:|:---------|
| age      |             29 |           77   |             0 |           120 | years            |             0 |             0 | pass     |
| trestbps |             94 |          200   |             1 |           300 | mm Hg            |             0 |             0 | pass     |
| chol     |            126 |          564   |             1 |           700 | mg/dl            |             0 |             0 | pass     |
| thalach  |             71 |          202   |             1 |           250 | beats per minute |             0 |             0 | pass     |
| oldpeak  |              0 |            6.2 |             0 |            10 | ST depression    |             0 |             0 | pass     |

## Encoded-Value Validation After Sentinel Decoding

| column   | observed_values   | valid_values   | sentinel_missing_values   | observed_sentinel_values   | unexpected_values   |   n_sentinel_rows |   n_unexpected_rows | status   |
|:---------|:------------------|:---------------|:--------------------------|:---------------------------|:--------------------|------------------:|--------------------:|:---------|
| sex      | 0, 1              | 0, 1           |                           |                            |                     |                 0 |                   0 | pass     |
| cp       | 0, 1, 2, 3        | 0, 1, 2, 3     |                           |                            |                     |                 0 |                   0 | pass     |
| fbs      | 0, 1              | 0, 1           |                           |                            |                     |                 0 |                   0 | pass     |
| restecg  | 0, 1, 2           | 0, 1, 2        |                           |                            |                     |                 0 |                   0 | pass     |
| exang    | 0, 1              | 0, 1           |                           |                            |                     |                 0 |                   0 | pass     |
| slope    | 0, 1, 2           | 0, 1, 2        |                           |                            |                     |                 0 |                   0 | pass     |
| ca       | 0, 1, 2, 3        | 0, 1, 2, 3     | 4                         |                            |                     |                 0 |                   0 | pass     |
| thal     | 1, 2, 3           | 1, 2, 3        | 0                         |                            |                     |                 0 |                   0 | pass     |
| target   | 0, 1              | 0, 1           |                           |                            |                     |                 0 |                   0 | pass     |

## Cleaning Decisions

- Rows after cleaning: 302
- Missing values introduced from sentinel decoding: 6
- Total missing values after cleaning: 6
- `trestbps = 200`, `chol = 564`, and high `oldpeak` values are retained because they are plausible clinical extremes, not invalid entries.
- `ca` and `thal` now contain missing values that will be handled by `SimpleImputer(strategy="most_frequent")` inside the preprocessing pipeline.
