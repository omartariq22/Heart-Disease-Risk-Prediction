"""Dataset loading and initial audit utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.schema import CATEGORICAL_COLUMNS, EXPECTED_COLUMNS, TARGET_COLUMN, TARGET_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
INITIAL_AUDIT_REPORT_PATH = REPORTS_DIR / "INITIAL_DATA_AUDIT.md"


@dataclass(frozen=True)
class InitialAudit:
    """Structured initial data-quality audit outputs."""

    shape: tuple[int, int]
    columns: list[str]
    dtypes: pd.Series
    missing_counts: pd.Series
    duplicate_count: int
    unique_values: pd.DataFrame
    descriptive_statistics: pd.DataFrame
    target_distribution: pd.DataFrame
    sample_records: pd.DataFrame


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw heart disease dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")

    return pd.read_csv(path)


def validate_expected_schema(df: pd.DataFrame) -> None:
    """Fail fast if the loaded dataset does not match the expected schema."""
    actual_columns = list(df.columns)
    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected dataset schema.\n"
            f"Expected columns: {EXPECTED_COLUMNS}\n"
            f"Actual columns:   {actual_columns}"
        )


def build_initial_audit(df: pd.DataFrame, sample_size: int = 5) -> InitialAudit:
    """Build the initial data audit required before cleaning or modelling."""
    validate_expected_schema(df)

    target_counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    target_distribution = (
        target_counts.rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
        .assign(
            label=lambda data: data[TARGET_COLUMN].map(TARGET_LABELS).fillna("Unknown"),
            percentage=lambda data: (data["count"] / len(df) * 100).round(2),
        )
    )

    unique_rows = []
    for column in CATEGORICAL_COLUMNS:
        values = sorted(df[column].dropna().unique().tolist())
        unique_rows.append(
            {
                "column": column,
                "n_unique": df[column].nunique(dropna=True),
                "unique_values": ", ".join(str(value) for value in values),
            }
        )

    return InitialAudit(
        shape=df.shape,
        columns=list(df.columns),
        dtypes=df.dtypes.astype(str),
        missing_counts=df.isna().sum(),
        duplicate_count=int(df.duplicated().sum()),
        unique_values=pd.DataFrame(unique_rows),
        descriptive_statistics=df.describe().round(2),
        target_distribution=target_distribution,
        sample_records=df.head(sample_size),
    )


def write_initial_audit_outputs(audit: InitialAudit) -> None:
    """Persist initial audit tables and a concise markdown report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    audit.dtypes.rename("dtype").to_frame().to_csv(RESULTS_DIR / "initial_dtypes.csv")
    audit.missing_counts.rename("missing_count").to_frame().to_csv(
        RESULTS_DIR / "initial_missing_values.csv"
    )
    audit.unique_values.to_csv(RESULTS_DIR / "initial_categorical_unique_values.csv", index=False)
    audit.descriptive_statistics.to_csv(RESULTS_DIR / "initial_descriptive_statistics.csv")
    audit.target_distribution.to_csv(RESULTS_DIR / "initial_target_distribution.csv", index=False)
    audit.sample_records.to_csv(RESULTS_DIR / "initial_sample_records.csv", index=False)

    report = render_initial_audit_markdown(audit)
    INITIAL_AUDIT_REPORT_PATH.write_text(report, encoding="utf-8")


def render_initial_audit_markdown(audit: InitialAudit) -> str:
    """Render the initial audit as a professional markdown report."""
    missing_total = int(audit.missing_counts.sum())
    target_counts = {
        int(row[TARGET_COLUMN]): int(row["count"])
        for _, row in audit.target_distribution.iterrows()
    }
    target_percentages = {
        int(row[TARGET_COLUMN]): float(row["percentage"])
        for _, row in audit.target_distribution.iterrows()
    }

    return f"""# Initial Data Audit

## Dataset Identity

- Source file: `data/raw/heart.csv`
- Rows: {audit.shape[0]}
- Columns: {audit.shape[1]}
- Target column: `target`
- Target meaning: `0` = no heart disease, `1` = heart disease present

## Schema Verification

The raw dataset was loaded with `pandas.read_csv` and validated against the expected 14-column schema. The column order matches the project contract:

```text
{", ".join(audit.columns)}
```

## Data Quality Summary

- Explicit missing values: {missing_total}
- Duplicate rows: {audit.duplicate_count}
- Target class counts: `0` = {target_counts.get(0, 0)}, `1` = {target_counts.get(1, 0)}
- Target class percentages: `0` = {target_percentages.get(0, 0.0):.2f}%, `1` = {target_percentages.get(1, 0.0):.2f}%

## Data Types

{audit.dtypes.rename("dtype").to_frame().to_markdown()}

## Missing Values By Column

{audit.missing_counts.rename("missing_count").to_frame().to_markdown()}

## Unique Values For Encoded Categorical Columns

{audit.unique_values.to_markdown(index=False)}

## Descriptive Statistics

{audit.descriptive_statistics.to_markdown()}

## Sample Records

{audit.sample_records.to_markdown(index=False)}

## Initial Observations

- The dataset contains 303 patient records and 14 variables, including the binary target.
- There are no explicit `NaN` values in the raw CSV.
- One duplicate row is present and must be removed during the cleaning step before train/test splitting.
- The target distribution is mildly imbalanced, with the positive class representing {target_percentages.get(1, 0.0):.2f}% of records.
- Encoded categorical columns are numeric in the CSV and require domain-aware validation before modelling.
- The sentinel-encoded missing-value issue for `ca = 4` and `thal = 0` will be handled in the cleaning step, not in this initial raw-data audit.
"""


def run_initial_audit() -> InitialAudit:
    """Load the raw dataset, build the audit, and persist all audit outputs."""
    df = load_raw_data()
    audit = build_initial_audit(df)
    write_initial_audit_outputs(audit)
    return audit


if __name__ == "__main__":
    run_initial_audit()
