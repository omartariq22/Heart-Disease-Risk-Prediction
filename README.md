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
