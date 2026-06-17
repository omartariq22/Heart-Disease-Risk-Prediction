"""Hyperparameter tuning for the strongest baseline models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import SEED
from src.data import load_raw_data
from src.model_preprocess import build_feature_target_split
from src.models import SCORING, build_model_pipeline, build_model_specs, evaluate_baseline_models
from src.preprocess import clean_heart_data
from src.schema import RESULTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
TUNING_REPORT_PATH = REPORTS_DIR / "HYPERPARAMETER_TUNING_REPORT.md"

TUNING_CANDIDATES = ["XGBoost", "Support Vector Machine", "Logistic Regression"]

PARAMETER_GRIDS = {
    "XGBoost": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.03, 0.05],
        "clf__max_depth": [2, 3],
    },
    "Support Vector Machine": {
        "clf__kernel": ["linear", "rbf"],
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", 0.1],
    },
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    },
}


@dataclass(frozen=True)
class TuningResult:
    """Structured hyperparameter tuning artifacts."""

    baseline_summary: pd.DataFrame
    split_summary: pd.DataFrame
    search_space: pd.DataFrame
    full_cv_results: pd.DataFrame
    tuned_comparison: pd.DataFrame
    selected_model: pd.DataFrame
    figure_paths: list[Path]


def _json_params(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def get_tuning_candidates(candidate_names: list[str] | None = None):
    """Return model specs selected for hyperparameter tuning."""
    names = candidate_names or TUNING_CANDIDATES
    return [spec for spec in build_model_specs() if spec.name in names]


def build_search_space_table(candidate_names: list[str] | None = None) -> pd.DataFrame:
    """Document each tuning grid."""
    rows = []
    for spec in get_tuning_candidates(candidate_names):
        grid = PARAMETER_GRIDS[spec.name]
        n_combinations = 1
        for values in grid.values():
            n_combinations *= len(values)
        rows.append(
            {
                "model": spec.name,
                "n_parameter_combinations": n_combinations,
                "parameter_grid": _json_params(grid),
            }
        )
    return pd.DataFrame(rows)


def _best_row_from_grid(model_name: str, grid: GridSearchCV) -> dict:
    best_index = int(grid.best_index_)
    cv_results = grid.cv_results_
    row = {
        "model": model_name,
        "best_params": _json_params(grid.best_params_),
        "best_refit_score": round(float(grid.best_score_), 6),
    }
    for metric in SCORING:
        row[f"{metric}_mean"] = round(float(cv_results[f"mean_test_{metric}"][best_index]), 6)
        row[f"{metric}_std"] = round(float(cv_results[f"std_test_{metric}"][best_index]), 6)
    return row


def _full_grid_results(model_name: str, grid: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(grid.cv_results_)
    keep_columns = ["rank_test_roc_auc", "params"]
    for metric in SCORING:
        keep_columns.extend([f"mean_test_{metric}", f"std_test_{metric}"])
    output = results[keep_columns].copy()
    output.insert(0, "model", model_name)
    output["params"] = output["params"].apply(_json_params)
    return output.sort_values(["model", "rank_test_roc_auc"]).reset_index(drop=True)


def tune_candidate_models(candidate_names: list[str] | None = None) -> TuningResult:
    """Tune selected models with GridSearchCV on the training split only."""
    raw = load_raw_data()
    cleaned = clean_heart_data(raw).cleaned
    feature_target_split = build_feature_target_split(cleaned)
    baseline_summary = evaluate_baseline_models().summary
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    best_rows = []
    full_results = []
    for spec in get_tuning_candidates(candidate_names):
        grid = GridSearchCV(
            estimator=build_model_pipeline(spec),
            param_grid=PARAMETER_GRIDS[spec.name],
            scoring=SCORING,
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        grid.fit(feature_target_split.x_train, feature_target_split.y_train)
        best_rows.append(_best_row_from_grid(spec.name, grid))
        full_results.append(_full_grid_results(spec.name, grid))

    tuned_comparison = (
        pd.DataFrame(best_rows)
        .sort_values(["recall_mean", "roc_auc_mean", "f1_mean"], ascending=False)
        .reset_index(drop=True)
    )
    selected_model = tuned_comparison.head(1).copy()
    selected_model.insert(1, "selection_reason", "Highest recall, then ROC-AUC and F1 among tuned candidates")

    figure_paths = generate_tuning_figures(tuned_comparison)

    return TuningResult(
        baseline_summary=baseline_summary,
        split_summary=feature_target_split.split.summary,
        search_space=build_search_space_table(candidate_names),
        full_cv_results=pd.concat(full_results, ignore_index=True),
        tuned_comparison=tuned_comparison,
        selected_model=selected_model,
        figure_paths=figure_paths,
    )


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def generate_tuning_figures(tuned_comparison: pd.DataFrame) -> list[Path]:
    """Generate tuning comparison figures."""
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    plot_data = tuned_comparison.melt(
        id_vars="model",
        value_vars=["recall_mean", "roc_auc_mean", "average_precision_mean", "f1_mean"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_data, x="model", y="score", hue="metric")
    plt.title("Tuned Candidate Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Cross-Validated Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    paths.append(_save_current_figure(FIGURES_DIR / "tuning_model_comparison.png"))

    return paths


def render_tuning_report(result: TuningResult) -> str:
    """Render the hyperparameter tuning report."""
    selected = result.selected_model.iloc[0]
    figure_list = "\n".join(f"- `outputs/figures/{path.name}`" for path in result.figure_paths)
    baseline_shortlist = result.baseline_summary.loc[
        result.baseline_summary["model"].isin(TUNING_CANDIDATES)
    ]

    return f"""# Hyperparameter Tuning Report

## Scope

This step tunes only the strongest baseline candidates: XGBoost, Support Vector Machine, and Logistic Regression. Grid search is performed on the training split only with stratified 5-fold cross-validation. The held-out test set is still not evaluated.

## Split Summary

{result.split_summary.to_markdown(index=False)}

## Search Space

{result.search_space.to_markdown(index=False)}

## Baseline Shortlist Metrics

{baseline_shortlist.to_markdown(index=False)}

## Tuned Model Comparison

Models are sorted by recall mean, then ROC-AUC mean, then F1 mean. GridSearchCV refits on ROC-AUC because threshold tuning handles recall-specific operating points.

{result.tuned_comparison.to_markdown(index=False)}

## Selected Tuned Candidate

{result.selected_model.to_markdown(index=False)}

## Full Grid Results

{result.full_cv_results.to_markdown(index=False)}

## Exported Figures

{figure_list}

## Decision

- Selected tuned candidate before final thresholding: {selected["model"]}.
- Best parameters: `{selected["best_params"]}`.
- Tuned recall: {selected["recall_mean"]:.3f}; tuned ROC-AUC: {selected["roc_auc_mean"]:.3f}; tuned F1: {selected["f1_mean"]:.3f}.
- The held-out test set remains untouched; final test evaluation should happen only after threshold choice is locked for the selected tuned model.
"""


def write_tuning_outputs(result: TuningResult) -> None:
    """Persist tuning tables, report, and figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    result.search_space.to_csv(RESULTS_DIR / "tuning_search_space.csv", index=False)
    result.full_cv_results.to_csv(RESULTS_DIR / "tuning_full_cv_results.csv", index=False)
    result.tuned_comparison.to_csv(RESULTS_DIR / "tuned_model_comparison.csv", index=False)
    result.selected_model.to_csv(RESULTS_DIR / "tuned_selected_model.csv", index=False)
    TUNING_REPORT_PATH.write_text(render_tuning_report(result), encoding="utf-8")


def run_hyperparameter_tuning() -> TuningResult:
    """Run Step 10 hyperparameter tuning and persist outputs."""
    result = tune_candidate_models()
    write_tuning_outputs(result)
    return result


if __name__ == "__main__":
    run_hyperparameter_tuning()
