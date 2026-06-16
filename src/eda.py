"""Exploratory data analysis for the heart disease training split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from src import SEED
from src.data import load_raw_data
from src.preprocess import clean_heart_data
from src.schema import (
    CATEGORICAL_COLUMNS,
    CHEST_PAIN_LABELS,
    EXERCISE_ANGINA_LABELS,
    EXPECTED_COLUMNS,
    NUMERICAL_COLUMNS,
    RESULTS_DIR,
    SEX_LABELS,
    TARGET_COLUMN,
    TARGET_LABELS,
    VALUE_LABELS,
)
from src.split import DataSplit, stratified_train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_REPORT_PATH = REPORTS_DIR / "EXPLORATORY_DATA_ANALYSIS.md"

PLOT_STYLE = "whitegrid"
PALETTE = ["#4C78A8", "#F58518"]


@dataclass(frozen=True)
class EDAResult:
    """Structured EDA artifacts for the training split."""

    split: DataSplit
    target_distribution: pd.DataFrame
    age_group_analysis: pd.DataFrame
    sex_analysis: pd.DataFrame
    numerical_summary_by_target: pd.DataFrame
    categorical_target_rates: pd.DataFrame
    pearson_correlation: pd.DataFrame
    spearman_correlation: pd.DataFrame
    chi_square_tests: pd.DataFrame
    mann_whitney_tests: pd.DataFrame
    mutual_information: pd.DataFrame
    figure_paths: list[Path]


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add age-group labels for EDA only."""
    enriched = df.copy()
    bins = [0, 39, 49, 59, 69, float("inf")]
    labels = ["<40", "40-49", "50-59", "60-69", "70+"]
    enriched["age_group"] = pd.cut(enriched["age"], bins=bins, labels=labels, right=True)
    return enriched


def target_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return class counts and percentages."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    return (
        counts.rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
        .assign(
            label=lambda data: data[TARGET_COLUMN].map(TARGET_LABELS),
            percentage=lambda data: (data["count"] / len(df) * 100).round(2),
        )
    )


def target_rate_table(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Calculate positive-target rate by a grouping column."""
    grouped = (
        df.groupby(group_column, observed=False)[TARGET_COLUMN]
        .agg(rows="count", heart_disease_cases="sum", heart_disease_rate="mean")
        .reset_index()
    )
    grouped["heart_disease_percentage"] = (grouped["heart_disease_rate"] * 100).round(2)
    return grouped.drop(columns=["heart_disease_rate"])


def sex_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate target distribution by sex with readable labels."""
    table = target_rate_table(df, "sex")
    table.insert(1, "sex_label", table["sex"].map(SEX_LABELS))
    return table


def categorical_target_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate positive-target rates for each categorical predictor."""
    rows = []
    for column in [c for c in CATEGORICAL_COLUMNS if c != TARGET_COLUMN]:
        grouped = target_rate_table(df, column)
        labels = VALUE_LABELS.get(column, {})
        for _, row in grouped.iterrows():
            value = row[column]
            rows.append(
                {
                    "feature": column,
                    "value": value,
                    "label": labels.get(value, str(value)),
                    "rows": int(row["rows"]),
                    "heart_disease_cases": int(row["heart_disease_cases"]),
                    "heart_disease_percentage": float(row["heart_disease_percentage"]),
                }
            )
    return pd.DataFrame(rows)


def numerical_summary_by_target(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize numerical distributions by target class."""
    rows = []
    grouped = df.groupby(TARGET_COLUMN)
    for target, group in grouped:
        for column in NUMERICAL_COLUMNS:
            series = group[column].dropna()
            rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "feature": column,
                    "mean": round(float(series.mean()), 3),
                    "median": round(float(series.median()), 3),
                    "std": round(float(series.std()), 3),
                    "min": round(float(series.min()), 3),
                    "max": round(float(series.max()), 3),
                }
            )
    return pd.DataFrame(rows)


def correlation_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Pearson and Spearman correlation matrices."""
    columns = NUMERICAL_COLUMNS + [TARGET_COLUMN]
    pearson = df[columns].corr(method="pearson").round(3)
    spearman = df[columns].corr(method="spearman").round(3)
    return pearson, spearman


def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run chi-square tests for categorical predictors versus target."""
    rows = []
    for column in [c for c in CATEGORICAL_COLUMNS if c != TARGET_COLUMN]:
        contingency = pd.crosstab(df[column], df[TARGET_COLUMN])
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        rows.append(
            {
                "feature": column,
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_value), 6),
                "degrees_of_freedom": int(dof),
                "significant_at_0_05": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def mann_whitney_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run Mann-Whitney U tests for numerical predictors versus target."""
    rows = []
    negative = df.loc[df[TARGET_COLUMN].eq(0)]
    positive = df.loc[df[TARGET_COLUMN].eq(1)]
    for column in NUMERICAL_COLUMNS:
        statistic, p_value = stats.mannwhitneyu(
            negative[column].dropna(),
            positive[column].dropna(),
            alternative="two-sided",
        )
        rows.append(
            {
                "feature": column,
                "mann_whitney_u": round(float(statistic), 4),
                "p_value": round(float(p_value), 6),
                "significant_at_0_05": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def mutual_information_table(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate feature relevance using mutual information."""
    features = [column for column in EXPECTED_COLUMNS if column != TARGET_COLUMN]
    x = df[features]
    y = df[TARGET_COLUMN]

    imputer = SimpleImputer(strategy="most_frequent")
    x_imputed = imputer.fit_transform(x)
    discrete_features = [column not in NUMERICAL_COLUMNS for column in features]
    scores = mutual_info_classif(
        x_imputed,
        y,
        discrete_features=discrete_features,
        random_state=SEED,
    )
    return (
        pd.DataFrame({"feature": features, "mutual_information": scores})
        .assign(mutual_information=lambda data: data["mutual_information"].round(6))
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def _target_label_series(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COLUMN].map(TARGET_LABELS)


def generate_eda_figures(
    train: pd.DataFrame,
    target_distribution: pd.DataFrame,
    age_group_analysis: pd.DataFrame,
    sex_analysis: pd.DataFrame,
    pearson_correlation: pd.DataFrame,
    mutual_information: pd.DataFrame,
) -> list[Path]:
    """Generate the Step 5 EDA figures."""
    sns.set_theme(style=PLOT_STYLE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    plot_df = train.copy()
    plot_df["target_label"] = _target_label_series(train)
    plot_df["sex_label"] = plot_df["sex"].map(SEX_LABELS)
    plot_df["cp_label"] = plot_df["cp"].map(CHEST_PAIN_LABELS)
    plot_df["exang_label"] = plot_df["exang"].map(EXERCISE_ANGINA_LABELS)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=target_distribution, x="label", y="count", hue="label", palette=PALETTE)
    plt.title("Target Class Distribution - Training Set")
    plt.xlabel("Target")
    plt.ylabel("Patient Count")
    plt.legend([], [], frameon=False)
    paths.append(_save_current_figure(FIGURES_DIR / "eda_target_distribution.png"))

    plt.figure(figsize=(7, 4))
    sns.histplot(data=plot_df, x="age", hue="target_label", bins=15, kde=True, palette=PALETTE)
    plt.title("Age Distribution By Target - Training Set")
    plt.xlabel("Age")
    plt.ylabel("Patient Count")
    paths.append(_save_current_figure(FIGURES_DIR / "eda_age_distribution_by_target.png"))

    plt.figure(figsize=(7, 4))
    sns.barplot(data=age_group_analysis, x="age_group", y="heart_disease_percentage", color=PALETTE[0])
    plt.title("Heart Disease Percentage By Age Group - Training Set")
    plt.xlabel("Age Group")
    plt.ylabel("Heart Disease (%)")
    paths.append(_save_current_figure(FIGURES_DIR / "eda_age_group_target_rate.png"))

    plt.figure(figsize=(6, 4))
    sns.barplot(data=sex_analysis, x="sex_label", y="heart_disease_percentage", color=PALETTE[1])
    plt.title("Heart Disease Percentage By Sex - Training Set")
    plt.xlabel("Sex")
    plt.ylabel("Heart Disease (%)")
    paths.append(_save_current_figure(FIGURES_DIR / "eda_sex_target_rate.png"))

    plt.figure(figsize=(7, 5))
    sns.heatmap(pearson_correlation, annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Pearson Correlation Heatmap - Training Set")
    paths.append(_save_current_figure(FIGURES_DIR / "eda_pearson_correlation_heatmap.png"))

    plt.figure(figsize=(9, 4))
    sns.countplot(data=plot_df, x="cp_label", hue="target_label", palette=PALETTE)
    plt.title("Chest Pain Type Versus Target - Training Set")
    plt.xlabel("Chest Pain Type")
    plt.ylabel("Patient Count")
    plt.xticks(rotation=25, ha="right")
    paths.append(_save_current_figure(FIGURES_DIR / "eda_chest_pain_vs_target.png"))

    plt.figure(figsize=(7, 4))
    sns.countplot(data=plot_df, x="exang_label", hue="target_label", palette=PALETTE)
    plt.title("Exercise-Induced Angina Versus Target - Training Set")
    plt.xlabel("Exercise-Induced Angina")
    plt.ylabel("Patient Count")
    paths.append(_save_current_figure(FIGURES_DIR / "eda_exang_vs_target.png"))

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=plot_df, x="target_label", y="thalach", hue="target_label", palette=PALETTE)
    plt.title("Maximum Heart Rate By Target - Training Set")
    plt.xlabel("Target")
    plt.ylabel("Maximum Heart Rate")
    plt.legend([], [], frameon=False)
    paths.append(_save_current_figure(FIGURES_DIR / "eda_thalach_by_target.png"))

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=plot_df, x="target_label", y="chol", hue="target_label", palette=PALETTE)
    plt.title("Cholesterol By Target - Training Set")
    plt.xlabel("Target")
    plt.ylabel("Serum Cholesterol")
    plt.legend([], [], frameon=False)
    paths.append(_save_current_figure(FIGURES_DIR / "eda_cholesterol_by_target.png"))

    plt.figure(figsize=(7, 5))
    top_mi = mutual_information.head(10).sort_values("mutual_information")
    sns.barplot(data=top_mi, x="mutual_information", y="feature", color=PALETTE[0])
    plt.title("Top Mutual Information Scores - Training Set")
    plt.xlabel("Mutual Information")
    plt.ylabel("Feature")
    paths.append(_save_current_figure(FIGURES_DIR / "eda_mutual_information.png"))

    return paths


def build_eda_result() -> EDAResult:
    """Build all EDA tables and figures from the cleaned training split."""
    raw = load_raw_data()
    cleaned = clean_heart_data(raw).cleaned
    split = stratified_train_test_split(cleaned)
    train = add_age_groups(split.train)

    target_dist = target_distribution_table(train)
    age_group_analysis = target_rate_table(train, "age_group")
    sex_analysis = sex_analysis_table(train)
    numeric_summary = numerical_summary_by_target(train)
    categorical_rates = categorical_target_rate_table(train)
    pearson, spearman = correlation_tables(train)
    chi_square = chi_square_tests(train)
    mann_whitney = mann_whitney_tests(train)
    mutual_info = mutual_information_table(train)
    figure_paths = generate_eda_figures(
        train=train,
        target_distribution=target_dist,
        age_group_analysis=age_group_analysis,
        sex_analysis=sex_analysis,
        pearson_correlation=pearson,
        mutual_information=mutual_info,
    )

    return EDAResult(
        split=split,
        target_distribution=target_dist,
        age_group_analysis=age_group_analysis,
        sex_analysis=sex_analysis,
        numerical_summary_by_target=numeric_summary,
        categorical_target_rates=categorical_rates,
        pearson_correlation=pearson,
        spearman_correlation=spearman,
        chi_square_tests=chi_square,
        mann_whitney_tests=mann_whitney,
        mutual_information=mutual_info,
        figure_paths=figure_paths,
    )


def render_eda_report(result: EDAResult) -> str:
    """Render the EDA report."""
    younger = result.age_group_analysis.loc[result.age_group_analysis["age_group"].eq("<40")]
    younger_rate = float(younger["heart_disease_percentage"].iloc[0]) if not younger.empty else 0.0
    sex_rates = result.sex_analysis.set_index("sex_label")["heart_disease_percentage"].to_dict()
    top_mi = result.mutual_information.head(5)
    significant_categorical = result.chi_square_tests.loc[
        result.chi_square_tests["significant_at_0_05"], "feature"
    ].tolist()
    significant_numeric = result.mann_whitney_tests.loc[
        result.mann_whitney_tests["significant_at_0_05"], "feature"
    ].tolist()
    figure_list = "\n".join(
        f"- `outputs/figures/{path.name}`" for path in result.figure_paths
    )

    return f"""# Exploratory Data Analysis

## Scope

This EDA is performed on the cleaned training split only. The held-out test set is not used for visual exploration, statistical testing, feature selection, or decision-making.

## Split Summary

{result.split.summary.to_markdown(index=False)}

## Target Distribution

{result.target_distribution.to_markdown(index=False)}

## Age-Based Analysis

{result.age_group_analysis.to_markdown(index=False)}

Younger patients (`<40`) have a heart disease-positive rate of {younger_rate:.2f}% in the training split.

## Sex-Based Analysis

{result.sex_analysis.to_markdown(index=False)}

Female positive rate: {sex_rates.get("Female", 0.0):.2f}%. Male positive rate: {sex_rates.get("Male", 0.0):.2f}%. In this dataset split, female patients show the higher positive-class percentage, while male patients are more numerous overall.

## Numerical Summary By Target

{result.numerical_summary_by_target.to_markdown(index=False)}

## Categorical Target Rates

{result.categorical_target_rates.to_markdown(index=False)}

## Pearson Correlation

{result.pearson_correlation.to_markdown()}

## Spearman Correlation

{result.spearman_correlation.to_markdown()}

## Chi-Square Tests For Categorical Predictors

{result.chi_square_tests.to_markdown(index=False)}

Significant categorical predictors at alpha = 0.05: {", ".join(significant_categorical) if significant_categorical else "None"}.

## Mann-Whitney U Tests For Numerical Predictors

{result.mann_whitney_tests.to_markdown(index=False)}

Significant numerical predictors at alpha = 0.05: {", ".join(significant_numeric) if significant_numeric else "None"}.

## Mutual Information

{result.mutual_information.to_markdown(index=False)}

Top mutual-information features:

{top_mi.to_markdown(index=False)}

## Exported Figures

{figure_list}

## Initial EDA Observations

- The training split preserves the mild class imbalance seen in the cleaned dataset.
- Age-group and sex-based target-rate tables directly answer the required project questions without using the test set.
- `thalach`, `oldpeak`, and several encoded clinical variables show meaningful target association and should be watched during modelling and interpretation.
- Statistical tests and mutual-information scores are exploratory evidence only; final feature importance must come from validated models in later steps.
"""


def write_eda_outputs(result: EDAResult) -> None:
    """Persist all EDA tables and report artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    result.split.summary.to_csv(RESULTS_DIR / "eda_split_summary.csv", index=False)
    result.target_distribution.to_csv(RESULTS_DIR / "eda_target_distribution.csv", index=False)
    result.age_group_analysis.to_csv(RESULTS_DIR / "eda_age_group_analysis.csv", index=False)
    result.sex_analysis.to_csv(RESULTS_DIR / "eda_sex_analysis.csv", index=False)
    result.numerical_summary_by_target.to_csv(
        RESULTS_DIR / "eda_numerical_summary_by_target.csv", index=False
    )
    result.categorical_target_rates.to_csv(
        RESULTS_DIR / "eda_categorical_target_rates.csv", index=False
    )
    result.pearson_correlation.to_csv(RESULTS_DIR / "eda_pearson_correlation.csv")
    result.spearman_correlation.to_csv(RESULTS_DIR / "eda_spearman_correlation.csv")
    result.chi_square_tests.to_csv(RESULTS_DIR / "eda_chi_square_tests.csv", index=False)
    result.mann_whitney_tests.to_csv(RESULTS_DIR / "eda_mann_whitney_tests.csv", index=False)
    result.mutual_information.to_csv(RESULTS_DIR / "eda_mutual_information.csv", index=False)
    EDA_REPORT_PATH.write_text(render_eda_report(result), encoding="utf-8")


def run_eda() -> EDAResult:
    """Run Step 5 exploratory data analysis and persist outputs."""
    result = build_eda_result()
    write_eda_outputs(result)
    return result


if __name__ == "__main__":
    run_eda()
