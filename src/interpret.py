"""Feature importance and model interpretation for tuned candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src import SEED
from src.data import load_raw_data
from src.model_preprocess import build_feature_target_split, build_preprocessor
from src.models import SklearnCompatibleXGBClassifier
from src.preprocess import clean_heart_data
from src.schema import RESULTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_IMPORTANCE_REPORT_PATH = REPORTS_DIR / "FEATURE_IMPORTANCE_REPORT.md"


@dataclass(frozen=True)
class InterpretationResult:
    """Structured feature-importance artifacts."""

    split_summary: pd.DataFrame
    selected_model_summary: pd.DataFrame
    svm_coefficients: pd.DataFrame
    logistic_coefficients: pd.DataFrame
    xgboost_importances: pd.DataFrame
    permutation_importance: pd.DataFrame
    insight_summary: pd.DataFrame
    figure_paths: list[Path]


def build_tuned_pipeline(model_name: str) -> Pipeline:
    """Build a tuned model pipeline using Step 10 selected configurations."""
    if model_name == "Support Vector Machine":
        estimator = SVC(
            C=0.1,
            gamma="scale",
            kernel="linear",
            probability=True,
            random_state=SEED,
        )
    elif model_name == "Logistic Regression":
        estimator = LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )
    elif model_name == "XGBoost":
        estimator = SklearnCompatibleXGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported tuned model for interpretation: {model_name}")

    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("clf", estimator),
        ]
    )


def _clean_feature_name(feature_name: str) -> str:
    return feature_name.replace("numeric__", "").replace("nominal__", "").replace(
        "ordinal__", ""
    ).replace("binary__", "")


def extract_linear_coefficients(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    """Extract coefficients from fitted linear classifiers."""
    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    coefficients = pipeline.named_steps["clf"].coef_.ravel()
    table = pd.DataFrame(
        {
            "model": model_name,
            "transformed_feature": feature_names,
            "feature": [_clean_feature_name(feature) for feature in feature_names],
            "coefficient": coefficients,
        }
    )
    table["abs_coefficient"] = table["coefficient"].abs()
    table["direction"] = table["coefficient"].apply(
        lambda value: "increases positive-class score" if value > 0 else "decreases positive-class score"
    )
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def extract_xgboost_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Extract fitted XGBoost feature importances."""
    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pipeline.named_steps["clf"].model_.feature_importances_
    table = pd.DataFrame(
        {
            "model": "XGBoost",
            "transformed_feature": feature_names,
            "feature": [_clean_feature_name(feature) for feature in feature_names],
            "importance": importances,
        }
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def calculate_permutation_importance(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Calculate model-agnostic permutation importance on the training split."""
    result = permutation_importance(
        pipeline,
        x_train,
        y_train,
        scoring="roc_auc",
        n_repeats=20,
        random_state=SEED,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "model": "Support Vector Machine",
            "feature": x_train.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def build_insight_summary(
    svm_coefficients: pd.DataFrame,
    logistic_coefficients: pd.DataFrame,
    xgboost_importances: pd.DataFrame,
    permutation_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize strongest predictors across interpretation methods."""
    rows = []
    rows.append(
        {
            "method": "Tuned SVM coefficients",
            "top_features": ", ".join(svm_coefficients["feature"].head(8).tolist()),
            "interpretation": "Largest absolute linear effects in the selected tuned model.",
        }
    )
    rows.append(
        {
            "method": "Tuned Logistic Regression coefficients",
            "top_features": ", ".join(logistic_coefficients["feature"].head(8).tolist()),
            "interpretation": "Linear coefficient sanity check from a strong calibrated baseline.",
        }
    )
    rows.append(
        {
            "method": "Tuned XGBoost importances",
            "top_features": ", ".join(xgboost_importances["feature"].head(8).tolist()),
            "interpretation": "Tree-based split/gain importance from the strongest boosted-tree candidate.",
        }
    )
    rows.append(
        {
            "method": "Permutation importance",
            "top_features": ", ".join(permutation_table["feature"].head(8).tolist()),
            "interpretation": "Model-agnostic ROC-AUC drop after shuffling raw input features.",
        }
    )
    return pd.DataFrame(rows)


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def generate_interpretation_figures(result: InterpretationResult) -> list[Path]:
    """Generate feature-importance figures."""
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    top_svm = result.svm_coefficients.head(12).sort_values("abs_coefficient")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_svm, x="coefficient", y="feature", hue="direction", dodge=False)
    plt.title("Top Tuned SVM Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Transformed Feature")
    paths.append(_save_current_figure(FIGURES_DIR / "interpret_svm_coefficients.png"))

    top_logistic = result.logistic_coefficients.head(12).sort_values("abs_coefficient")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_logistic, x="coefficient", y="feature", hue="direction", dodge=False)
    plt.title("Top Tuned Logistic Regression Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Transformed Feature")
    paths.append(_save_current_figure(FIGURES_DIR / "interpret_logistic_coefficients.png"))

    top_xgb = result.xgboost_importances.head(12).sort_values("importance")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_xgb, x="importance", y="feature", color="#4C78A8")
    plt.title("Top Tuned XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Transformed Feature")
    paths.append(_save_current_figure(FIGURES_DIR / "interpret_xgboost_importances.png"))

    top_permutation = result.permutation_importance.head(12).sort_values("importance_mean")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_permutation, x="importance_mean", y="feature", color="#F58518")
    plt.title("Permutation Importance For Tuned SVM")
    plt.xlabel("Mean ROC-AUC Drop")
    plt.ylabel("Raw Feature")
    paths.append(_save_current_figure(FIGURES_DIR / "interpret_permutation_importance.png"))

    return paths


def build_interpretation_result() -> InterpretationResult:
    """Fit tuned models on training data and build interpretation artifacts."""
    raw = load_raw_data()
    cleaned = clean_heart_data(raw).cleaned
    feature_target_split = build_feature_target_split(cleaned)

    svm_pipeline = build_tuned_pipeline("Support Vector Machine")
    logistic_pipeline = build_tuned_pipeline("Logistic Regression")
    xgb_pipeline = build_tuned_pipeline("XGBoost")

    svm_pipeline.fit(feature_target_split.x_train, feature_target_split.y_train)
    logistic_pipeline.fit(feature_target_split.x_train, feature_target_split.y_train)
    xgb_pipeline.fit(feature_target_split.x_train, feature_target_split.y_train)

    svm_coefficients = extract_linear_coefficients(svm_pipeline, "Support Vector Machine")
    logistic_coefficients = extract_linear_coefficients(logistic_pipeline, "Logistic Regression")
    xgboost_importances = extract_xgboost_importances(xgb_pipeline)
    permutation_table = calculate_permutation_importance(
        svm_pipeline,
        feature_target_split.x_train,
        feature_target_split.y_train,
    )
    insight_summary = build_insight_summary(
        svm_coefficients,
        logistic_coefficients,
        xgboost_importances,
        permutation_table,
    )
    selected_model_summary = pd.DataFrame(
        [
            {
                "selected_model": "Support Vector Machine",
                "best_params": '{"clf__C": 0.1, "clf__gamma": "scale", "clf__kernel": "linear"}',
                "interpretation_basis": "Tuned linear SVM coefficients plus permutation importance.",
            }
        ]
    )

    preliminary = InterpretationResult(
        split_summary=feature_target_split.split.summary,
        selected_model_summary=selected_model_summary,
        svm_coefficients=svm_coefficients,
        logistic_coefficients=logistic_coefficients,
        xgboost_importances=xgboost_importances,
        permutation_importance=permutation_table,
        insight_summary=insight_summary,
        figure_paths=[],
    )
    figure_paths = generate_interpretation_figures(preliminary)

    return InterpretationResult(
        split_summary=preliminary.split_summary,
        selected_model_summary=preliminary.selected_model_summary,
        svm_coefficients=svm_coefficients,
        logistic_coefficients=logistic_coefficients,
        xgboost_importances=xgboost_importances,
        permutation_importance=permutation_table,
        insight_summary=insight_summary,
        figure_paths=figure_paths,
    )


def render_interpretation_report(result: InterpretationResult) -> str:
    """Render the feature-importance report."""
    figure_list = "\n".join(f"- `outputs/figures/{path.name}`" for path in result.figure_paths)

    return f"""# Feature Importance And Medical Insight Report

## Scope

This interpretation step fits tuned candidate pipelines on the training split only. The held-out test set remains untouched. Interpretability results are educational and exploratory; they do not establish clinical causality.

## Split Summary

{result.split_summary.to_markdown(index=False)}

## Selected Model For Interpretation

{result.selected_model_summary.to_markdown(index=False)}

## Cross-Method Insight Summary

{result.insight_summary.to_markdown(index=False)}

## Tuned SVM Coefficients

Positive coefficients increase the model score for heart disease presence; negative coefficients decrease it.

{result.svm_coefficients.to_markdown(index=False)}

## Tuned Logistic Regression Coefficients

{result.logistic_coefficients.to_markdown(index=False)}

## Tuned XGBoost Feature Importances

{result.xgboost_importances.to_markdown(index=False)}

## Tuned SVM Permutation Importance

{result.permutation_importance.to_markdown(index=False)}

## Exported Figures

{figure_list}

## Project Question Connections

- Chest pain type, thalassemia category, exercise-induced angina, vessel count, `oldpeak`, and maximum heart rate repeatedly appear as important signals across model families.
- Sex appears in the linear models and EDA as a meaningful association in this dataset, but this should be framed as dataset-specific rather than a universal medical conclusion.
- Age has weaker model importance than several exercise/ECG-related variables, even though age-group EDA remains useful for answering the project's demographic questions.
- The strongest model signals align with clinically plausible cardiac stress and diagnostic markers, but the dataset is small and historical, so conclusions remain educational.
"""


def write_interpretation_outputs(result: InterpretationResult) -> None:
    """Persist interpretation tables, figures, and report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    result.svm_coefficients.to_csv(RESULTS_DIR / "interpret_svm_coefficients.csv", index=False)
    result.logistic_coefficients.to_csv(
        RESULTS_DIR / "interpret_logistic_coefficients.csv", index=False
    )
    result.xgboost_importances.to_csv(
        RESULTS_DIR / "interpret_xgboost_importances.csv", index=False
    )
    result.permutation_importance.to_csv(
        RESULTS_DIR / "interpret_permutation_importance.csv", index=False
    )
    result.insight_summary.to_csv(RESULTS_DIR / "interpret_insight_summary.csv", index=False)
    FEATURE_IMPORTANCE_REPORT_PATH.write_text(render_interpretation_report(result), encoding="utf-8")


def run_feature_interpretation() -> InterpretationResult:
    """Run Step 11 feature interpretation and persist outputs."""
    result = build_interpretation_result()
    write_interpretation_outputs(result)
    return result


if __name__ == "__main__":
    run_feature_interpretation()
