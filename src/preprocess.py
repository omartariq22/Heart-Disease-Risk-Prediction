"""Data cleaning utilities for the heart disease dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_raw_data, validate_expected_schema
from src.schema import (
    CATEGORICAL_COLUMNS,
    EXPECTED_COLUMNS,
    NUMERICAL_COLUMNS,
    RESULTS_DIR,
    SENTINEL_MISSING_VALUES,
    TARGET_COLUMN,
    validate_encoded_values,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "heart_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
CLEANING_REPORT_PATH = REPORTS_DIR / "DATA_CLEANING_REPORT.md"

NUMERIC_RANGE_RULES = {
    "age": {"min": 0, "max": 120, "unit": "years"},
    "trestbps": {"min": 1, "max": 300, "unit": "mm Hg"},
    "chol": {"min": 1, "max": 700, "unit": "mg/dl"},
    "thalach": {"min": 1, "max": 250, "unit": "beats per minute"},
    "oldpeak": {"min": 0, "max": 10, "unit": "ST depression"},
}


@dataclass(frozen=True)
class CleaningResult:
    """Structured result of the deterministic cleaning step."""

    raw: pd.DataFrame
    cleaned: pd.DataFrame
    duplicate_records: pd.DataFrame
    sentinel_replacements: pd.DataFrame
    missing_values: pd.DataFrame
    numeric_range_validation: pd.DataFrame
    categorical_validation: pd.DataFrame
    summary: pd.DataFrame


def find_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate records with their original row indices preserved."""
    duplicate_records = df.loc[df.duplicated(keep=False)].copy()
    duplicate_records.insert(0, "original_index", duplicate_records.index)
    return duplicate_records


def drop_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows before any train/test split."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def decode_sentinel_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert documented sentinel-coded missing values to NaN."""
    cleaned = df.copy()
    rows = []

    for column, sentinel_values in SENTINEL_MISSING_VALUES.items():
        mask = cleaned[column].isin(sentinel_values)
        n_replaced = int(mask.sum())
        rows.append(
            {
                "column": column,
                "sentinel_values": ", ".join(str(value) for value in sorted(sentinel_values)),
                "n_replaced_with_nan": n_replaced,
            }
        )
        cleaned.loc[mask, column] = np.nan

    return cleaned, pd.DataFrame(rows)


def build_missing_values_table(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Compare missingness before and after deterministic cleaning."""
    return pd.DataFrame(
        {
            "column": EXPECTED_COLUMNS,
            "raw_missing_count": raw[EXPECTED_COLUMNS].isna().sum().values,
            "cleaned_missing_count": cleaned[EXPECTED_COLUMNS].isna().sum().values,
        }
    )


def validate_numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate broad clinical plausibility ranges for numeric features."""
    rows = []
    for column in NUMERICAL_COLUMNS:
        rule = NUMERIC_RANGE_RULES[column]
        series = df[column].dropna()
        n_below_min = int((series < rule["min"]).sum())
        n_above_max = int((series > rule["max"]).sum())
        rows.append(
            {
                "column": column,
                "observed_min": float(series.min()),
                "observed_max": float(series.max()),
                "allowed_min": rule["min"],
                "allowed_max": rule["max"],
                "unit": rule["unit"],
                "n_below_min": n_below_min,
                "n_above_max": n_above_max,
                "status": "pass" if n_below_min == 0 and n_above_max == 0 else "review",
            }
        )
    return pd.DataFrame(rows)


def build_cleaning_summary(
    raw: pd.DataFrame,
    deduplicated: pd.DataFrame,
    cleaned: pd.DataFrame,
    sentinel_replacements: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact cleaning summary table."""
    target_counts = cleaned[TARGET_COLUMN].value_counts().sort_index()
    return pd.DataFrame(
        [
            {"metric": "raw_rows", "value": len(raw)},
            {"metric": "raw_columns", "value": raw.shape[1]},
            {"metric": "duplicate_rows_removed", "value": len(raw) - len(deduplicated)},
            {"metric": "cleaned_rows", "value": len(cleaned)},
            {"metric": "cleaned_columns", "value": cleaned.shape[1]},
            {"metric": "raw_explicit_missing_values", "value": int(raw.isna().sum().sum())},
            {"metric": "cleaned_missing_values", "value": int(cleaned.isna().sum().sum())},
            {
                "metric": "sentinel_values_decoded_to_nan",
                "value": int(sentinel_replacements["n_replaced_with_nan"].sum()),
            },
            {"metric": "target_0_count_after_cleaning", "value": int(target_counts.get(0, 0))},
            {"metric": "target_1_count_after_cleaning", "value": int(target_counts.get(1, 0))},
        ]
    )


def clean_heart_data(raw: pd.DataFrame) -> CleaningResult:
    """Apply deterministic cleaning without fitting any model-time imputers."""
    validate_expected_schema(raw)

    duplicate_records = find_duplicate_records(raw)
    deduplicated = drop_duplicate_records(raw)
    cleaned, sentinel_replacements = decode_sentinel_missing_values(deduplicated)

    missing_values = build_missing_values_table(raw=raw, cleaned=cleaned)
    numeric_range_validation = validate_numeric_ranges(cleaned)
    categorical_validation = validate_encoded_values(cleaned)
    summary = build_cleaning_summary(raw, deduplicated, cleaned, sentinel_replacements)

    return CleaningResult(
        raw=raw,
        cleaned=cleaned,
        duplicate_records=duplicate_records,
        sentinel_replacements=sentinel_replacements,
        missing_values=missing_values,
        numeric_range_validation=numeric_range_validation,
        categorical_validation=categorical_validation,
        summary=summary,
    )


def render_cleaning_report(result: CleaningResult) -> str:
    """Render a concise professional data-cleaning report."""
    summary_lookup = dict(zip(result.summary["metric"], result.summary["value"]))
    sentinel_total = int(summary_lookup["sentinel_values_decoded_to_nan"])
    cleaned_missing = int(summary_lookup["cleaned_missing_values"])

    return f"""# Data Cleaning Report

## Purpose

This report documents the deterministic data-cleaning actions applied after the raw-data audit and schema validation. The cleaned snapshot is generated for inspection and reporting; modelling code should still read from `data/raw/heart.csv`, call the same cleaning function, and perform imputation inside the modelling pipeline to avoid leakage.

## Cleaning Actions

- Dropped exact duplicate rows before any train/test split.
- Decoded documented sentinel missing values to `NaN`:
  - `ca = 4`
  - `thal = 0`
- Preserved medically plausible clinical extremes for later outlier analysis.
- Did not impute missing values in the dataframe; imputation belongs inside the modelling pipeline.

## Cleaning Summary

{result.summary.to_markdown(index=False)}

## Duplicate Records

{result.duplicate_records.to_markdown(index=False)}

## Sentinel Replacements

{result.sentinel_replacements.to_markdown(index=False)}

## Missing Values After Cleaning

{result.missing_values.to_markdown(index=False)}

## Numeric Range Validation

{result.numeric_range_validation.to_markdown(index=False)}

## Encoded-Value Validation After Sentinel Decoding

{result.categorical_validation.to_markdown(index=False)}

## Cleaning Decisions

- Rows after cleaning: {summary_lookup["cleaned_rows"]}
- Missing values introduced from sentinel decoding: {sentinel_total}
- Total missing values after cleaning: {cleaned_missing}
- `trestbps = 200`, `chol = 564`, and high `oldpeak` values are retained because they are plausible clinical extremes, not invalid entries.
- `ca` and `thal` now contain missing values that will be handled by `SimpleImputer(strategy="most_frequent")` inside the preprocessing pipeline.
"""


def write_cleaning_outputs(result: CleaningResult) -> None:
    """Persist cleaned inspection data, validation tables, and report."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    result.cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    result.summary.to_csv(RESULTS_DIR / "cleaning_summary.csv", index=False)
    result.duplicate_records.to_csv(RESULTS_DIR / "duplicate_records.csv", index=False)
    result.sentinel_replacements.to_csv(RESULTS_DIR / "sentinel_replacements.csv", index=False)
    result.missing_values.to_csv(RESULTS_DIR / "post_clean_missing_values.csv", index=False)
    result.numeric_range_validation.to_csv(
        RESULTS_DIR / "numeric_range_validation.csv", index=False
    )
    result.categorical_validation.to_csv(
        RESULTS_DIR / "post_clean_categorical_value_validation.csv", index=False
    )
    CLEANING_REPORT_PATH.write_text(render_cleaning_report(result), encoding="utf-8")


def run_data_cleaning() -> CleaningResult:
    """Load raw data, apply deterministic cleaning, and write Step 4 outputs."""
    raw = load_raw_data()
    result = clean_heart_data(raw)
    write_cleaning_outputs(result)
    return result


if __name__ == "__main__":
    run_data_cleaning()
