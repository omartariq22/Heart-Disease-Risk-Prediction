# Final Test Evaluation Report

## Scope

This report evaluates the locked final model once on the held-out 20% test set. Model selection, hyperparameter tuning, and threshold selection were completed before this evaluation.

## Locked Decision

- Model: Support Vector Machine
- Best parameters: `{"clf__C": 0.1, "clf__gamma": "scale", "clf__kernel": "linear"}`
- Locked operating point: recall_at_least_0.90
- Locked threshold: 0.40
- Cross-validated recall before final test: 0.908
- Cross-validated ROC-AUC before final test: 0.921
- Cross-validated F1 before final test: 0.876

## Split Summary

| split   |   rows |   target_0_count |   target_1_count |   target_1_percentage |
|:--------|-------:|-----------------:|-----------------:|----------------------:|
| train   |    241 |              110 |              131 |                 54.36 |
| test    |     61 |               28 |               33 |                 54.1  |

## Held-Out Test Metrics

| model                  | evaluation_split   | operating_point      |   threshold |   accuracy |   precision |   recall |   specificity |       f1 |   roc_auc |   average_precision |   true_negative |   false_positive |   false_negative |   true_positive |
|:-----------------------|:-------------------|:---------------------|------------:|-----------:|------------:|---------:|--------------:|---------:|----------:|--------------------:|----------------:|-----------------:|-----------------:|----------------:|
| Support Vector Machine | held_out_test      | default_0.50         |         0.5 |   0.803279 |    0.769231 | 0.909091 |      0.678571 | 0.833333 |  0.912338 |            0.926877 |              19 |                9 |                3 |              30 |
| Support Vector Machine | held_out_test      | recall_at_least_0.90 |         0.4 |   0.770492 |    0.731707 | 0.909091 |      0.607143 | 0.810811 |  0.912338 |            0.926877 |              17 |               11 |                3 |              30 |

## Model Artifact

The fitted final pipeline is saved locally at `outputs/models/final_model.joblib`. This binary artifact is intentionally ignored by Git because it is generated output.

## Interpretation

The locked threshold should be interpreted as a recall-priority operating point. In this project context, false negatives are more serious than false positives because a false negative represents a heart-disease-positive patient predicted as negative. These results remain educational and should not be used for clinical decision-making.
