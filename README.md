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

Run tests:

```powershell
pytest
```
