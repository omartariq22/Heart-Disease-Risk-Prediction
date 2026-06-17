# Heart Disease Risk Prediction

A professional data mining and machine learning project for heart disease prediction using structured preprocessing, exploratory analysis, outlier detection, model comparison, threshold tuning, and reproducible reporting.

## Runtime

- Python 3.11+
- Pinned dependencies in `requirements.txt`
- Global reproducibility seed: `src.SEED = 42`

## Project Structure

```text
data/
  raw/                 Original input data; never modified manually.
  processed/           Generated cleaned snapshots.
notebooks/             Thin narrative notebooks that call reusable code.
src/                   Reusable project logic.
outputs/
  figures/             Exported 300 dpi plots.
  models/              Serialized trained pipelines.
  results/             CSV result tables.
reports/               Report assets and model card.
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The raw dataset is stored at `data/raw/heart.csv`.

## Current Workflow

Generate the initial raw-data audit:

```powershell
python -m src.data
```

This writes machine-readable audit tables to `outputs/results/` and a human-readable report to `reports/INITIAL_DATA_AUDIT.md`.

Generate the formal data dictionary and encoded-value validation:

```powershell
python -m src.schema
```

This writes schema tables to `outputs/results/` and the data dictionary report to `reports/DATA_DICTIONARY.md`.

Run deterministic data cleaning:

```powershell
python -m src.preprocess
```

This writes a cleaned inspection snapshot to `data/processed/heart_clean.csv`, cleaning tables to `outputs/results/`, and the cleaning report to `reports/DATA_CLEANING_REPORT.md`.

Run exploratory data analysis:

```powershell
python -m src.eda
```

This creates training-only EDA result tables in `outputs/results/`, 300 dpi figures in `outputs/figures/`, and the EDA report at `reports/EXPLORATORY_DATA_ANALYSIS.md`.

Run outlier detection:

```powershell
python -m src.outliers
```

This creates training-only IQR/z-score outlier tables in `outputs/results/`, diagnostic figures in `outputs/figures/`, and the outlier report at `reports/OUTLIER_DETECTION_REPORT.md`.

Inspect the shared modelling preprocessor:

```powershell
python -m src.model_preprocess
```

This writes preprocessing inspection tables to `outputs/results/` and the preprocessing report to `reports/PREPROCESSING_PIPELINE_REPORT.md`.

Run baseline model comparison:

```powershell
python -m src.models
```

This evaluates the candidate classifiers with stratified 5-fold cross-validation on the training split only, writes `outputs/results/cv_results.csv`, and creates `reports/BASELINE_MODEL_COMPARISON.md`.

Run model evaluation:

```powershell
python -m src.evaluate
```

This uses out-of-fold training predictions to generate ROC/PR curves, default-threshold metrics, threshold sweeps, confusion matrices, and `reports/MODEL_EVALUATION_REPORT.md`. The held-out test set is still not evaluated at this stage.

Run hyperparameter tuning:

```powershell
python -m src.tune
```

This tunes the strongest baseline candidates with stratified 5-fold GridSearchCV, writes tuning tables to `outputs/results/`, and creates `reports/HYPERPARAMETER_TUNING_REPORT.md`.

Run feature interpretation:

```powershell
python -m src.interpret
```

This fits tuned candidate pipelines on the training split, writes coefficient/importance tables to `outputs/results/`, exports interpretation figures, and creates `reports/FEATURE_IMPORTANCE_REPORT.md`.

Run visualization QA:

```powershell
python -m src.plots
```

This indexes all generated figures, creates the final CV comparison and calibration plots, writes visualization QA tables to `outputs/results/`, and creates `reports/VISUALIZATION_REPORT.md`.

Run tests:

```powershell
pytest
```
