# Model Card - Heart Disease Risk Prediction

## Model Details

- Final model: Support Vector Machine
- Pipeline artifact: `outputs/models/final_model.joblib`
- Preprocessing: median imputation and scaling for numeric features, most-frequent imputation and one-hot encoding for nominal features, most-frequent imputation for `ca`, passthrough for binary features
- Random seed: 42
- Locked threshold: 0.40 (recall_at_least_0.90)

## Intended Use

This model is intended for an educational data mining project that demonstrates supervised classification, preprocessing, model comparison, threshold tuning, and interpretation on a small heart disease dataset.

## Not Intended Use

This model must not be used for diagnosis, triage, screening, treatment decisions, or patient-facing medical recommendations.

## Training Data

The model uses the processed Cleveland heart disease dataset supplied in `data/raw/heart.csv`. After duplicate removal, the dataset contains 302 rows. The final pipeline is fitted on the stratified training split with 241 rows and evaluated once on the held-out test split with 61 rows.

## Metrics

### Cross-Validated Training Metrics For Selected Model

- Recall: 0.908
- ROC-AUC: 0.921
- F1: 0.876

### Held-Out Test Metrics

| operating_point | threshold | accuracy | precision | recall | specificity | f1 | roc_auc | average_precision | false_negative | false_positive |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| default_0.50 | 0.50 | 0.803 | 0.769 | 0.909 | 0.679 | 0.833 | 0.912 | 0.927 | 3 | 9 |
| recall_at_least_0.90 | 0.40 | 0.770 | 0.732 | 0.909 | 0.607 | 0.811 | 0.912 | 0.927 | 3 | 11 |

## Ethical And Practical Limitations

- The data is small, historical, and from a limited clinical context.
- The target is a simplified binary label derived from a richer disease-severity variable.
- Dataset-specific sex and age patterns should not be generalized to real populations.
- The model is not calibrated or validated for clinical deployment.
- Predictions are sensitive to the dataset's encoding, preprocessing assumptions, and threshold choice.

## Recommended Governance

Use this model only as a reproducible coursework artifact. Any real medical use would require modern representative data, clinical validation, calibration assessment, bias analysis, monitoring, and review by qualified medical professionals.
