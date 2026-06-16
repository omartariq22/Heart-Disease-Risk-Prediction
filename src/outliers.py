"""Outlier detection for numerical clinical features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import load_raw_data
from src.preprocess import clean_heart_data
from src.schema import NUMERICAL_COLUMNS, RESULTS_DIR, TARGET_COLUMN, TARGET_LABELS
from src.split import DataSplit, stratified_train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTLIER_REPORT_PATH = REPORTS_DIR / "OUTLIER_DETECTION_REPORT.md"

PALETTE = ["#4C78A8", "#F58518"]
ZSCORE_THRESHOLD = 3.0


@dataclass(frozen=True)
class OutlierResult:
    """Structured outputs for outlier detection."""

    split: DataSplit
    iqr_summary: pd.DataFrame
    iqr_records: pd.DataFrame
    zscore_summary: pd.DataFrame
    zscore_records: pd.DataFrame
    treatment_recommendations: pd.DataFrame
    figure_paths: list[Path]


def calculate_iqr_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outliers using the 1.5 * IQR rule."""
    summary_rows = []
    record_rows = []

    for feature in NUMERICAL_COLUMNS:
        series = df[feature].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = df[feature].lt(lower_bound) | df[feature].gt(upper_bound)
        outliers = df.loc[mask, [feature, TARGET_COLUMN]].copy()

        summary_rows.append(
            {
                "feature": feature,
                "q1": round(q1, 3),
                "q3": round(q3, 3),
                "iqr": round(iqr, 3),
                "lower_bound": round(lower_bound, 3),
                "upper_bound": round(upper_bound, 3),
                "outlier_count": int(mask.sum()),
                "outlier_percentage": round(float(mask.mean() * 100), 2),
                "min_outlier": round(float(outliers[feature].min()), 3) if not outliers.empty else np.nan,
                "max_outlier": round(float(outliers[feature].max()), 3) if not outliers.empty else np.nan,
            }
        )

        for index, row in outliers.iterrows():
            record_rows.append(
                {
                    "row_index_in_training_split": int(index),
                    "feature": feature,
                    "value": float(row[feature]),
                    "target": int(row[TARGET_COLUMN]),
                    "target_label": TARGET_LABELS[int(row[TARGET_COLUMN])],
                    "lower_bound": round(lower_bound, 3),
                    "upper_bound": round(upper_bound, 3),
                    "method": "iqr",
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(record_rows)


def calculate_zscore_outliers(
    df: pd.DataFrame,
    threshold: float = ZSCORE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect outliers using absolute z-scores."""
    summary_rows = []
    record_rows = []

    for feature in NUMERICAL_COLUMNS:
        series = df[feature].dropna()
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        zscores = (df[feature] - mean) / std if std > 0 else pd.Series(0.0, index=df.index)
        mask = zscores.abs().gt(threshold)
        outliers = df.loc[mask, [feature, TARGET_COLUMN]].copy()

        summary_rows.append(
            {
                "feature": feature,
                "mean": round(mean, 3),
                "std": round(std, 3),
                "threshold": threshold,
                "outlier_count": int(mask.sum()),
                "outlier_percentage": round(float(mask.mean() * 100), 2),
                "min_outlier": round(float(outliers[feature].min()), 3) if not outliers.empty else np.nan,
                "max_outlier": round(float(outliers[feature].max()), 3) if not outliers.empty else np.nan,
            }
        )

        for index, row in outliers.iterrows():
            record_rows.append(
                {
                    "row_index_in_training_split": int(index),
                    "feature": feature,
                    "value": float(row[feature]),
                    "target": int(row[TARGET_COLUMN]),
                    "target_label": TARGET_LABELS[int(row[TARGET_COLUMN])],
                    "zscore": round(float(zscores.loc[index]), 3),
                    "threshold": threshold,
                    "method": "zscore",
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(record_rows)


def build_treatment_recommendations(iqr_summary: pd.DataFrame) -> pd.DataFrame:
    """Recommend whether to keep, cap, or investigate detected outliers."""
    rows = []
    for _, row in iqr_summary.iterrows():
        feature = row["feature"]
        outlier_count = int(row["outlier_count"])

        if outlier_count == 0:
            recommendation = "No action"
            rationale = "No IQR outliers were detected in the training split."
        else:
            recommendation = "Keep for modelling"
            rationale = (
                "Detected values are clinically plausible extremes in this dataset. "
                "Document them and let robust validation decide model behavior."
            )

        rows.append(
            {
                "feature": feature,
                "iqr_outlier_count": outlier_count,
                "recommendation": recommendation,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def generate_outlier_figures(train: pd.DataFrame) -> list[Path]:
    """Generate visual outlier diagnostics."""
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    plot_df = train.copy()
    plot_df["target_label"] = plot_df[TARGET_COLUMN].map(TARGET_LABELS)

    fig, axes = plt.subplots(len(NUMERICAL_COLUMNS), 1, figsize=(8, 12))
    for axis, feature in zip(axes, NUMERICAL_COLUMNS):
        sns.boxplot(data=plot_df, x=feature, ax=axis, color="#4C78A8")
        axis.set_title(f"{feature} Box Plot - Training Set")
        axis.set_xlabel(feature)
    paths.append(_save_current_figure(FIGURES_DIR / "outliers_numeric_boxplots.png"))

    for feature in NUMERICAL_COLUMNS:
        plt.figure(figsize=(7, 4))
        sns.histplot(
            data=plot_df,
            x=feature,
            hue="target_label",
            kde=True,
            bins=20,
            palette=PALETTE,
        )
        plt.title(f"{feature} Distribution By Target - Training Set")
        plt.xlabel(feature)
        plt.ylabel("Patient Count")
        paths.append(_save_current_figure(FIGURES_DIR / f"outliers_{feature}_histogram_kde.png"))

    fig, axes = plt.subplots(1, len(NUMERICAL_COLUMNS), figsize=(18, 4))
    for axis, feature in zip(axes, NUMERICAL_COLUMNS):
        sns.boxplot(data=plot_df, y=feature, x="target_label", hue="target_label", ax=axis, palette=PALETTE)
        axis.set_title(feature)
        axis.set_xlabel("Target")
        axis.set_ylabel(feature)
        axis.legend([], [], frameon=False)
    paths.append(_save_current_figure(FIGURES_DIR / "outliers_boxplots_by_target.png"))

    return paths


def build_outlier_result() -> OutlierResult:
    """Build all Step 6 outlier artifacts from the cleaned training split."""
    raw = load_raw_data()
    cleaned = clean_heart_data(raw).cleaned
    split = stratified_train_test_split(cleaned)

    iqr_summary, iqr_records = calculate_iqr_outliers(split.train)
    zscore_summary, zscore_records = calculate_zscore_outliers(split.train)
    treatment_recommendations = build_treatment_recommendations(iqr_summary)
    figure_paths = generate_outlier_figures(split.train)

    return OutlierResult(
        split=split,
        iqr_summary=iqr_summary,
        iqr_records=iqr_records,
        zscore_summary=zscore_summary,
        zscore_records=zscore_records,
        treatment_recommendations=treatment_recommendations,
        figure_paths=figure_paths,
    )


def render_outlier_report(result: OutlierResult) -> str:
    """Render the outlier detection report."""
    iqr_features = result.iqr_summary.loc[
        result.iqr_summary["outlier_count"].gt(0), "feature"
    ].tolist()
    zscore_features = result.zscore_summary.loc[
        result.zscore_summary["outlier_count"].gt(0), "feature"
    ].tolist()
    figure_list = "\n".join(f"- `outputs/figures/{path.name}`" for path in result.figure_paths)

    iqr_records_md = (
        result.iqr_records.to_markdown(index=False)
        if not result.iqr_records.empty
        else "No IQR outlier records detected."
    )
    zscore_records_md = (
        result.zscore_records.to_markdown(index=False)
        if not result.zscore_records.empty
        else "No z-score outlier records detected."
    )

    return f"""# Outlier Detection Report

## Scope

Outlier detection is performed on the cleaned training split only. The held-out test split remains untouched and is not used to make preprocessing or modelling decisions.

## Split Summary

{result.split.summary.to_markdown(index=False)}

## IQR Outlier Summary

{result.iqr_summary.to_markdown(index=False)}

Features with IQR outliers: {", ".join(iqr_features) if iqr_features else "None"}.

## IQR Outlier Records

{iqr_records_md}

## Z-Score Outlier Summary

{result.zscore_summary.to_markdown(index=False)}

Features with absolute z-score > {ZSCORE_THRESHOLD}: {", ".join(zscore_features) if zscore_features else "None"}.

## Z-Score Outlier Records

{zscore_records_md}

## Treatment Recommendations

{result.treatment_recommendations.to_markdown(index=False)}

## Exported Figures

{figure_list}

## Decision

- No records are removed in Step 6.
- Detected outliers are retained because the values are clinically plausible and were already validated against broad medical plausibility ranges during cleaning.
- If a future model is strongly distorted by extreme values, the professional next step is to compare cross-validated performance with and without winsorization, not to delete observations ad hoc.
- Imputation and scaling remain inside the modelling pipeline to avoid leakage.
"""


def write_outlier_outputs(result: OutlierResult) -> None:
    """Persist outlier tables, figures, and report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    result.iqr_summary.to_csv(RESULTS_DIR / "outlier_iqr_summary.csv", index=False)
    result.iqr_records.to_csv(RESULTS_DIR / "outlier_iqr_records.csv", index=False)
    result.zscore_summary.to_csv(RESULTS_DIR / "outlier_zscore_summary.csv", index=False)
    result.zscore_records.to_csv(RESULTS_DIR / "outlier_zscore_records.csv", index=False)
    result.treatment_recommendations.to_csv(
        RESULTS_DIR / "outlier_treatment_recommendations.csv", index=False
    )
    OUTLIER_REPORT_PATH.write_text(render_outlier_report(result), encoding="utf-8")


def run_outlier_detection() -> OutlierResult:
    """Run Step 6 outlier detection and persist outputs."""
    result = build_outlier_result()
    write_outlier_outputs(result)
    return result


if __name__ == "__main__":
    run_outlier_detection()
