# Reproducibility Check Report

## Scope

This report records the final end-to-end execution order for the Heart Disease Risk Prediction project. The workflow regenerates all audit tables, cleaning outputs, EDA and outlier artifacts, model-selection outputs, final test metrics, reports, and the local model artifact from the raw dataset.

## Execution Summary

- Workflow steps executed: 14
- Workflow status: passed
- Total measured workflow time: 40.420 seconds
- Test suite status: passed

|   order | step                        | status   |   elapsed_seconds | error_message   |
|--------:|:----------------------------|:---------|------------------:|:----------------|
|       1 | initial_data_audit          | passed   |             0.039 |                 |
|       2 | schema_validation           | passed   |             0.007 |                 |
|       3 | data_cleaning               | passed   |             0.014 |                 |
|       4 | exploratory_data_analysis   | passed   |             1.546 |                 |
|       5 | outlier_detection           | passed   |             1.997 |                 |
|       6 | preprocessing_inspection    | passed   |             0.026 |                 |
|       7 | baseline_model_comparison   | passed   |             2.723 |                 |
|       8 | model_evaluation            | passed   |             7.488 |                 |
|       9 | hyperparameter_tuning       | passed   |            23.35  |                 |
|      10 | feature_interpretation      | passed   |             2.157 |                 |
|      11 | visualization_qa            | passed   |             0.965 |                 |
|      12 | final_model_package         | passed   |             0.053 |                 |
|      13 | consolidated_reporting      | passed   |             0.048 |                 |
|      14 | submission_acceptance_check | passed   |             0.007 |                 |

## Notes

- Run command: `python -m src.run_all`
- Use `python -m src.run_all --skip-tests` to regenerate outputs without running `pytest`.
- The final serialized model is generated at `outputs/models/final_model.joblib` and is intentionally ignored by Git.
- The reproducibility contract uses the fixed global seed `src.SEED = 42`.
