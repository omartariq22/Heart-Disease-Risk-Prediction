# Outlier Detection Report

## Scope

Outlier detection is performed on the cleaned training split only. The held-out test split remains untouched and is not used to make preprocessing or modelling decisions.

## Split Summary

| split   |   rows |   target_0_count |   target_1_count |   target_1_percentage |
|:--------|-------:|-----------------:|-----------------:|----------------------:|
| train   |    241 |              110 |              131 |                 54.36 |
| test    |     61 |               28 |               33 |                 54.1  |

## IQR Outlier Summary

| feature   |   q1 |    q3 |   iqr |   lower_bound |   upper_bound |   outlier_count |   outlier_percentage |   min_outlier |   max_outlier |
|:----------|-----:|------:|------:|--------------:|--------------:|----------------:|---------------------:|--------------:|--------------:|
| age       |   47 |  61   |  14   |          26   |          82   |               0 |                 0    |         nan   |         nan   |
| trestbps  |  120 | 140   |  20   |          90   |         170   |               9 |                 3.73 |         172   |         200   |
| chol      |  211 | 273   |  62   |         118   |         366   |               3 |                 1.24 |         394   |         409   |
| thalach   |  136 | 168   |  32   |          88   |         216   |               0 |                 0    |         nan   |         nan   |
| oldpeak   |    0 |   1.8 |   1.8 |          -2.7 |           4.5 |               2 |                 0.83 |           5.6 |           6.2 |

Features with IQR outliers: trestbps, chol, oldpeak.

## IQR Outlier Records

|   row_index_in_training_split | feature   |   value |   target | target_label          |   lower_bound |   upper_bound | method   |
|------------------------------:|:----------|--------:|---------:|:----------------------|--------------:|--------------:|:---------|
|                             6 | trestbps  |   172   |        1 | Heart disease present |          90   |         170   | iqr      |
|                            80 | trestbps  |   178   |        1 | Heart disease present |          90   |         170   | iqr      |
|                            87 | trestbps  |   180   |        1 | Heart disease present |          90   |         170   | iqr      |
|                           160 | trestbps  |   180   |        0 | No heart disease      |          90   |         170   | iqr      |
|                           179 | trestbps  |   200   |        0 | No heart disease      |          90   |         170   | iqr      |
|                           192 | trestbps  |   174   |        0 | No heart disease      |          90   |         170   | iqr      |
|                           199 | trestbps  |   192   |        0 | No heart disease      |          90   |         170   | iqr      |
|                           209 | trestbps  |   178   |        0 | No heart disease      |          90   |         170   | iqr      |
|                           214 | trestbps  |   180   |        0 | No heart disease      |          90   |         170   | iqr      |
|                            76 | chol      |   394   |        1 | Heart disease present |         118   |         366   | iqr      |
|                           176 | chol      |   407   |        0 | No heart disease      |         118   |         366   | iqr      |
|                           197 | chol      |   409   |        0 | No heart disease      |         118   |         366   | iqr      |
|                           161 | oldpeak   |     6.2 |        0 | No heart disease      |          -2.7 |           4.5 | iqr      |
|                           177 | oldpeak   |     5.6 |        0 | No heart disease      |          -2.7 |           4.5 | iqr      |

## Z-Score Outlier Summary

| feature   |    mean |    std |   threshold |   outlier_count |   outlier_percentage |   min_outlier |   max_outlier |
|:----------|--------:|-------:|------------:|----------------:|---------------------:|--------------:|--------------:|
| age       |  54.34  |  9.201 |           3 |               0 |                 0    |         nan   |         nan   |
| trestbps  | 131.448 | 17.951 |           3 |               2 |                 0.83 |         192   |         200   |
| chol      | 244.299 | 48.006 |           3 |               3 |                 1.24 |         394   |         409   |
| thalach   | 150.071 | 22.041 |           3 |               0 |                 0    |         nan   |         nan   |
| oldpeak   |   1.071 |  1.204 |           3 |               2 |                 0.83 |           5.6 |           6.2 |

Features with absolute z-score > 3.0: trestbps, chol, oldpeak.

## Z-Score Outlier Records

|   row_index_in_training_split | feature   |   value |   target | target_label          |   zscore |   threshold | method   |
|------------------------------:|:----------|--------:|---------:|:----------------------|---------:|------------:|:---------|
|                           179 | trestbps  |   200   |        0 | No heart disease      |    3.819 |           3 | zscore   |
|                           199 | trestbps  |   192   |        0 | No heart disease      |    3.373 |           3 | zscore   |
|                            76 | chol      |   394   |        1 | Heart disease present |    3.118 |           3 | zscore   |
|                           176 | chol      |   407   |        0 | No heart disease      |    3.389 |           3 | zscore   |
|                           197 | chol      |   409   |        0 | No heart disease      |    3.431 |           3 | zscore   |
|                           161 | oldpeak   |     6.2 |        0 | No heart disease      |    4.261 |           3 | zscore   |
|                           177 | oldpeak   |     5.6 |        0 | No heart disease      |    3.762 |           3 | zscore   |

## Treatment Recommendations

| feature   |   iqr_outlier_count | recommendation     | rationale                                                                                                                         |
|:----------|--------------------:|:-------------------|:----------------------------------------------------------------------------------------------------------------------------------|
| age       |                   0 | No action          | No IQR outliers were detected in the training split.                                                                              |
| trestbps  |                   9 | Keep for modelling | Detected values are clinically plausible extremes in this dataset. Document them and let robust validation decide model behavior. |
| chol      |                   3 | Keep for modelling | Detected values are clinically plausible extremes in this dataset. Document them and let robust validation decide model behavior. |
| thalach   |                   0 | No action          | No IQR outliers were detected in the training split.                                                                              |
| oldpeak   |                   2 | Keep for modelling | Detected values are clinically plausible extremes in this dataset. Document them and let robust validation decide model behavior. |

## Exported Figures

- `outputs/figures/outliers_numeric_boxplots.png`
- `outputs/figures/outliers_age_histogram_kde.png`
- `outputs/figures/outliers_trestbps_histogram_kde.png`
- `outputs/figures/outliers_chol_histogram_kde.png`
- `outputs/figures/outliers_thalach_histogram_kde.png`
- `outputs/figures/outliers_oldpeak_histogram_kde.png`
- `outputs/figures/outliers_boxplots_by_target.png`

## Decision

- No records are removed in Step 6.
- Detected outliers are retained because the values are clinically plausible and were already validated against broad medical plausibility ranges during cleaning.
- If a future model is strongly distorted by extreme values, the professional next step is to compare cross-validated performance with and without winsorization, not to delete observations ad hoc.
- Imputation and scaling remain inside the modelling pipeline to avoid leakage.
