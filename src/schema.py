"""Dataset schema, feature groups, and encoded-value validation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DICTIONARY_REPORT_PATH = REPORTS_DIR / "DATA_DICTIONARY.md"

TARGET_COLUMN = "target"

EXPECTED_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

NUMERICAL_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_COLUMNS = ["sex", "fbs", "exang", "target"]
NOMINAL_CATEGORICAL_COLUMNS = ["cp", "restecg", "slope", "thal"]
ORDINAL_COUNT_COLUMNS = ["ca"]
CATEGORICAL_COLUMNS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
    "target",
]

FEATURE_GROUPS = {
    "numerical": NUMERICAL_COLUMNS,
    "binary": BINARY_COLUMNS,
    "nominal_categorical": NOMINAL_CATEGORICAL_COLUMNS,
    "ordinal_count": ORDINAL_COUNT_COLUMNS,
}

SEX_LABELS = {
    0: "Female",
    1: "Male",
}

CHEST_PAIN_LABELS = {
    0: "Typical angina",
    1: "Atypical angina",
    2: "Non-anginal pain",
    3: "Asymptomatic",
}

FASTING_BLOOD_SUGAR_LABELS = {
    0: "Fasting blood sugar <= 120 mg/dl",
    1: "Fasting blood sugar > 120 mg/dl",
}

REST_ECG_LABELS = {
    0: "Normal",
    1: "ST-T wave abnormality",
    2: "Probable/definite left ventricular hypertrophy",
}

EXERCISE_ANGINA_LABELS = {
    0: "No exercise-induced angina",
    1: "Exercise-induced angina",
}

SLOPE_LABELS = {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping",
}

CA_LABELS = {
    0: "0 major vessels",
    1: "1 major vessel",
    2: "2 major vessels",
    3: "3 major vessels",
}

THAL_LABELS = {
    1: "Fixed defect",
    2: "Normal",
    3: "Reversible defect",
}

TARGET_LABELS = {
    0: "No heart disease",
    1: "Heart disease present",
}

VALUE_LABELS: dict[str, Mapping[int, str]] = {
    "sex": SEX_LABELS,
    "cp": CHEST_PAIN_LABELS,
    "fbs": FASTING_BLOOD_SUGAR_LABELS,
    "restecg": REST_ECG_LABELS,
    "exang": EXERCISE_ANGINA_LABELS,
    "slope": SLOPE_LABELS,
    "ca": CA_LABELS,
    "thal": THAL_LABELS,
    "target": TARGET_LABELS,
}

VALID_ENCODED_VALUES = {
    "sex": {0, 1},
    "cp": {0, 1, 2, 3},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3},
    "thal": {1, 2, 3},
    "target": {0, 1},
}

SENTINEL_MISSING_VALUES = {
    "ca": {4},
    "thal": {0},
}


def _format_labels(labels: Mapping[int, str]) -> str:
    return "; ".join(f"{key} = {value}" for key, value in labels.items())


def _format_values(values: set[int]) -> str:
    return ", ".join(str(value) for value in sorted(values))


def build_data_dictionary() -> pd.DataFrame:
    """Return the formal data dictionary for the processed Cleveland subset."""
    rows = [
        {
            "column": "age",
            "description": "Patient age in years.",
            "feature_group": "numerical",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": "Positive adult age in years",
            "sentinel_missing_values": "",
            "labels": "",
        },
        {
            "column": "sex",
            "description": "Patient sex encoded as a binary category.",
            "feature_group": "binary",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": _format_values(VALID_ENCODED_VALUES["sex"]),
            "sentinel_missing_values": "",
            "labels": _format_labels(SEX_LABELS),
        },
        {
            "column": "cp",
            "description": "Chest pain type.",
            "feature_group": "nominal_categorical",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": _format_values(VALID_ENCODED_VALUES["cp"]),
            "sentinel_missing_values": "",
            "labels": _format_labels(CHEST_PAIN_LABELS),
        },
        {
            "column": "trestbps",
            "description": "Resting blood pressure in mm Hg on admission.",
            "feature_group": "numerical",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": "Positive mm Hg value",
            "sentinel_missing_values": "",
            "labels": "",
        },
        {
            "column": "chol",
            "description": "Serum cholesterol in mg/dl.",
            "feature_group": "numerical",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": "Positive mg/dl value",
            "sentinel_missing_values": "",
            "labels": "",
        },
        {
            "column": "fbs",
            "description": "Whether fasting blood sugar is greater than 120 mg/dl.",
            "feature_group": "binary",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": _format_values(VALID_ENCODED_VALUES["fbs"]),
            "sentinel_missing_values": "",
            "labels": _format_labels(FASTING_BLOOD_SUGAR_LABELS),
        },
        {
            "column": "restecg",
            "description": "Resting electrocardiographic result.",
            "feature_group": "nominal_categorical",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": _format_values(VALID_ENCODED_VALUES["restecg"]),
            "sentinel_missing_values": "",
            "labels": _format_labels(REST_ECG_LABELS),
        },
        {
            "column": "thalach",
            "description": "Maximum heart rate achieved.",
            "feature_group": "numerical",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": "Positive beats-per-minute value",
            "sentinel_missing_values": "",
            "labels": "",
        },
        {
            "column": "exang",
            "description": "Whether exercise-induced angina is present.",
            "feature_group": "binary",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": _format_values(VALID_ENCODED_VALUES["exang"]),
            "sentinel_missing_values": "",
            "labels": _format_labels(EXERCISE_ANGINA_LABELS),
        },
        {
            "column": "oldpeak",
            "description": "ST depression induced by exercise relative to rest.",
            "feature_group": "numerical",
            "expected_dtype": "float",
            "modeling_role": "Predictor",
            "valid_values": "Non-negative continuous value",
            "sentinel_missing_values": "",
            "labels": "",
        },
        {
            "column": "slope",
            "description": "Slope of the peak exercise ST segment.",
            "feature_group": "nominal_categorical",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": _format_values(VALID_ENCODED_VALUES["slope"]),
            "sentinel_missing_values": "",
            "labels": _format_labels(SLOPE_LABELS),
        },
        {
            "column": "ca",
            "description": "Number of major vessels colored by fluoroscopy.",
            "feature_group": "ordinal_count",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": _format_values(VALID_ENCODED_VALUES["ca"]),
            "sentinel_missing_values": _format_values(SENTINEL_MISSING_VALUES["ca"]),
            "labels": _format_labels(CA_LABELS),
        },
        {
            "column": "thal",
            "description": "Thalassemia category.",
            "feature_group": "nominal_categorical",
            "expected_dtype": "integer",
            "modeling_role": "Predictor",
            "valid_values": _format_values(VALID_ENCODED_VALUES["thal"]),
            "sentinel_missing_values": _format_values(SENTINEL_MISSING_VALUES["thal"]),
            "labels": _format_labels(THAL_LABELS),
        },
        {
            "column": "target",
            "description": "Binary heart disease diagnosis label.",
            "feature_group": "binary",
            "expected_dtype": "integer",
            "modeling_role": "Target",
            "valid_values": _format_values(VALID_ENCODED_VALUES["target"]),
            "sentinel_missing_values": "",
            "labels": _format_labels(TARGET_LABELS),
        },
    ]

    data_dictionary = pd.DataFrame(rows)
    data_dictionary["column"] = pd.Categorical(
        data_dictionary["column"], categories=EXPECTED_COLUMNS, ordered=True
    )
    return data_dictionary.sort_values("column").reset_index(drop=True)


def build_feature_groups_table() -> pd.DataFrame:
    """Return feature groups as a compact table for reports."""
    rows = []
    for group_name, columns in FEATURE_GROUPS.items():
        rows.append(
            {
                "feature_group": group_name,
                "columns": ", ".join(columns),
                "n_columns": len(columns),
            }
        )
    return pd.DataFrame(rows)


def validate_encoded_values(df: pd.DataFrame) -> pd.DataFrame:
    """Validate observed categorical encodings against the data dictionary."""
    rows = []
    for column in CATEGORICAL_COLUMNS:
        observed = set(df[column].dropna().unique().tolist())
        valid = VALID_ENCODED_VALUES[column]
        sentinels = SENTINEL_MISSING_VALUES.get(column, set())
        sentinel_observed = observed.intersection(sentinels)
        unexpected = observed.difference(valid).difference(sentinels)

        if unexpected:
            status = "fail"
        elif sentinel_observed:
            status = "sentinel_missing_detected"
        else:
            status = "pass"

        rows.append(
            {
                "column": column,
                "observed_values": _format_values(observed),
                "valid_values": _format_values(valid),
                "sentinel_missing_values": _format_values(sentinels),
                "observed_sentinel_values": _format_values(sentinel_observed),
                "unexpected_values": _format_values(unexpected),
                "n_sentinel_rows": int(df[column].isin(sentinels).sum()) if sentinels else 0,
                "n_unexpected_rows": int(df[column].isin(unexpected).sum()) if unexpected else 0,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def render_data_dictionary_markdown(
    data_dictionary: pd.DataFrame,
    feature_groups: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    """Render the data dictionary and encoded-value validation report."""
    sentinel_rows = validation.loc[validation["status"].eq("sentinel_missing_detected")]
    failed_rows = validation.loc[validation["status"].eq("fail")]

    if failed_rows.empty:
        validation_summary = (
            "No unexpected categorical encodings were found. Sentinel-coded missing values "
            "were detected and will be converted to `NaN` during Step 4 cleaning."
        )
    else:
        validation_summary = (
            "Unexpected categorical encodings were found. These must be investigated before modelling."
        )

    return f"""# Data Dictionary And Schema Validation

## Purpose

This document defines the authoritative schema for the heart disease prediction project. It separates true numerical features from encoded categorical variables, records human-readable labels for reporting, and validates observed categorical values before cleaning and modelling.

## Feature Groups

{feature_groups.to_markdown(index=False)}

## Data Dictionary

{data_dictionary.to_markdown(index=False)}

## Encoded-Value Validation

{validation.to_markdown(index=False)}

## Validation Summary

- {validation_summary}
- Columns with sentinel-coded missing values: {", ".join(sentinel_rows["column"].tolist()) if not sentinel_rows.empty else "None"}
- Columns with unexpected invalid values: {", ".join(failed_rows["column"].tolist()) if not failed_rows.empty else "None"}

## Modelling Notes

- Numerical features will be imputed with the median and scaled in the modelling pipeline.
- Nominal categorical features will be imputed with the most frequent value and one-hot encoded in the modelling pipeline.
- `ca` is treated as an ordinal/count feature after sentinel decoding because values 0-3 represent ordered vessel counts.
- Binary variables are passed through after validation because their encodings are already machine-readable.
- `ca = 4` and `thal = 0` are not valid medical categories for this project schema; they are documented sentinel values and must be decoded to missing values in Step 4.
"""


def write_schema_outputs(df: pd.DataFrame) -> None:
    """Persist the data dictionary, feature groups, validation table, and report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data_dictionary = build_data_dictionary()
    feature_groups = build_feature_groups_table()
    validation = validate_encoded_values(df)

    data_dictionary.to_csv(RESULTS_DIR / "data_dictionary.csv", index=False)
    feature_groups.to_csv(RESULTS_DIR / "feature_groups.csv", index=False)
    validation.to_csv(RESULTS_DIR / "categorical_value_validation.csv", index=False)

    report = render_data_dictionary_markdown(data_dictionary, feature_groups, validation)
    DATA_DICTIONARY_REPORT_PATH.write_text(report, encoding="utf-8")


def run_schema_validation() -> None:
    """Load the raw dataset and write Step 3 schema artifacts."""
    from src.data import load_raw_data, validate_expected_schema

    df = load_raw_data()
    validate_expected_schema(df)
    write_schema_outputs(df)


if __name__ == "__main__":
    run_schema_validation()
