# Initial Data Audit

## Dataset Identity

- Source file: `data/raw/heart.csv`
- Rows: 303
- Columns: 14
- Target column: `target`
- Target meaning: `0` = no heart disease, `1` = heart disease present

## Schema Verification

The raw dataset was loaded with `pandas.read_csv` and validated against the expected 14-column schema. The column order matches the project contract:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
```

## Data Quality Summary

- Explicit missing values: 0
- Duplicate rows: 1
- Target class counts: `0` = 138, `1` = 165
- Target class percentages: `0` = 45.54%, `1` = 54.46%

## Data Types

|          | dtype   |
|:---------|:--------|
| age      | int64   |
| sex      | int64   |
| cp       | int64   |
| trestbps | int64   |
| chol     | int64   |
| fbs      | int64   |
| restecg  | int64   |
| thalach  | int64   |
| exang    | int64   |
| oldpeak  | float64 |
| slope    | int64   |
| ca       | int64   |
| thal     | int64   |
| target   | int64   |

## Missing Values By Column

|          |   missing_count |
|:---------|----------------:|
| age      |               0 |
| sex      |               0 |
| cp       |               0 |
| trestbps |               0 |
| chol     |               0 |
| fbs      |               0 |
| restecg  |               0 |
| thalach  |               0 |
| exang    |               0 |
| oldpeak  |               0 |
| slope    |               0 |
| ca       |               0 |
| thal     |               0 |
| target   |               0 |

## Unique Values For Encoded Categorical Columns

| column   |   n_unique | unique_values   |
|:---------|-----------:|:----------------|
| sex      |          2 | 0, 1            |
| cp       |          4 | 0, 1, 2, 3      |
| fbs      |          2 | 0, 1            |
| restecg  |          3 | 0, 1, 2         |
| exang    |          2 | 0, 1            |
| slope    |          3 | 0, 1, 2         |
| ca       |          5 | 0, 1, 2, 3, 4   |
| thal     |          4 | 0, 1, 2, 3      |
| target   |          2 | 0, 1            |

## Descriptive Statistics

|       |    age |    sex |     cp |   trestbps |   chol |    fbs |   restecg |   thalach |   exang |   oldpeak |   slope |     ca |   thal |   target |
|:------|-------:|-------:|-------:|-----------:|-------:|-------:|----------:|----------:|--------:|----------:|--------:|-------:|-------:|---------:|
| count | 303    | 303    | 303    |     303    | 303    | 303    |    303    |    303    |  303    |    303    |  303    | 303    | 303    |   303    |
| mean  |  54.37 |   0.68 |   0.97 |     131.62 | 246.26 |   0.15 |      0.53 |    149.65 |    0.33 |      1.04 |    1.4  |   0.73 |   2.31 |     0.54 |
| std   |   9.08 |   0.47 |   1.03 |      17.54 |  51.83 |   0.36 |      0.53 |     22.91 |    0.47 |      1.16 |    0.62 |   1.02 |   0.61 |     0.5  |
| min   |  29    |   0    |   0    |      94    | 126    |   0    |      0    |     71    |    0    |      0    |    0    |   0    |   0    |     0    |
| 25%   |  47.5  |   0    |   0    |     120    | 211    |   0    |      0    |    133.5  |    0    |      0    |    1    |   0    |   2    |     0    |
| 50%   |  55    |   1    |   1    |     130    | 240    |   0    |      1    |    153    |    0    |      0.8  |    1    |   0    |   2    |     1    |
| 75%   |  61    |   1    |   2    |     140    | 274.5  |   0    |      1    |    166    |    1    |      1.6  |    2    |   1    |   3    |     1    |
| max   |  77    |   1    |   3    |     200    | 564    |   1    |      2    |    202    |    1    |      6.2  |    2    |   4    |   3    |     1    |

## Sample Records

|   age |   sex |   cp |   trestbps |   chol |   fbs |   restecg |   thalach |   exang |   oldpeak |   slope |   ca |   thal |   target |
|------:|------:|-----:|-----------:|-------:|------:|----------:|----------:|--------:|----------:|--------:|-----:|-------:|---------:|
|    63 |     1 |    3 |        145 |    233 |     1 |         0 |       150 |       0 |       2.3 |       0 |    0 |      1 |        1 |
|    37 |     1 |    2 |        130 |    250 |     0 |         1 |       187 |       0 |       3.5 |       0 |    0 |      2 |        1 |
|    41 |     0 |    1 |        130 |    204 |     0 |         0 |       172 |       0 |       1.4 |       2 |    0 |      2 |        1 |
|    56 |     1 |    1 |        120 |    236 |     0 |         1 |       178 |       0 |       0.8 |       2 |    0 |      2 |        1 |
|    57 |     0 |    0 |        120 |    354 |     0 |         1 |       163 |       1 |       0.6 |       2 |    0 |      2 |        1 |

## Initial Observations

- The dataset contains 303 patient records and 14 variables, including the binary target.
- There are no explicit `NaN` values in the raw CSV.
- One duplicate row is present and must be removed during the cleaning step before train/test splitting.
- The target distribution is mildly imbalanced, with the positive class representing 54.46% of records.
- Encoded categorical columns are numeric in the CSV and require domain-aware validation before modelling.
- The sentinel-encoded missing-value issue for `ca = 4` and `thal = 0` will be handled in the cleaning step, not in this initial raw-data audit.
