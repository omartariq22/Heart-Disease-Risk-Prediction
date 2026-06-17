# Reproducibility Check Report

## Scope

This report records the final end-to-end execution order for the Heart Disease Risk Prediction project. The workflow regenerates all audit tables, cleaning outputs, EDA and outlier artifacts, model-selection outputs, final test metrics, reports, and the local model artifact from the raw dataset.

## Execution Summary

- Workflow steps executed: 13
- Workflow status: passed
- Total measured workflow time: 22.484 seconds
- Test suite status: passed

|   order | step                      | status   |   elapsed_seconds | error_message   |
|--------:|:--------------------------|:---------|------------------:|:----------------|
|       1 | initial_data_audit        | passed   |             0.041 |                 |
|       2 | schema_validation         | passed   |             0.008 |                 |
|       3 | data_cleaning             | passed   |             0.013 |                 |
|       4 | exploratory_data_analysis | passed   |             1.476 |                 |
|       5 | outlier_detection         | passed   |             1.961 |                 |
|       6 | preprocessing_inspection  | passed   |             0.024 |                 |
|       7 | baseline_model_comparison | passed   |             2.678 |                 |
|       8 | model_evaluation          | passed   |             6.772 |                 |
|       9 | hyperparameter_tuning     | passed   |             6.429 |                 |
|      10 | feature_interpretation    | passed   |             2.079 |                 |
|      11 | visualization_qa          | passed   |             0.917 |                 |
|      12 | final_model_package       | passed   |             0.041 |                 |
|      13 | consolidated_reporting    | passed   |             0.045 |                 |

## Notes

- Run command: `python -m src.run_all`
- Use `python -m src.run_all --skip-tests` to regenerate outputs without running `pytest`.
- The final serialized model is generated at `outputs/models/final_model.joblib` and is intentionally ignored by Git.
- The reproducibility contract uses the fixed global seed `src.SEED = 42`.
