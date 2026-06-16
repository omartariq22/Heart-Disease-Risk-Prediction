"""Baseline model lineup and cross-validation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import SEED
from src.data import load_raw_data
from src.model_preprocess import build_feature_target_split, build_preprocessor
from src.preprocess import clean_heart_data
from src.schema import RESULTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_MODEL_REPORT_PATH = REPORTS_DIR / "BASELINE_MODEL_COMPARISON.md"

SCORING = {
    "accuracy": get_scorer("accuracy"),
    "precision": get_scorer("precision"),
    "recall": get_scorer("recall"),
    "f1": get_scorer("f1"),
    "roc_auc": get_scorer("roc_auc"),
    "average_precision": get_scorer("average_precision"),
}


@dataclass(frozen=True)
class ModelSpec:
    """Classifier metadata used for baseline comparison."""

    name: str
    estimator: BaseEstimator
    notes: str


class SklearnCompatibleXGBClassifier(ClassifierMixin, BaseEstimator):
    """Small sklearn-compatible adapter for XGBClassifier under sklearn 1.6."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        eval_metric: str = "logloss",
        random_state: int = SEED,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, x, y):
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(x, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, x):
        return self.model_.predict(x)

    def predict_proba(self, x):
        return self.model_.predict_proba(x)


@dataclass(frozen=True)
class BaselineModelResult:
    """Structured baseline model comparison outputs."""

    model_inventory: pd.DataFrame
    fold_results: pd.DataFrame
    summary: pd.DataFrame
    split_summary: pd.DataFrame


def build_model_specs() -> list[ModelSpec]:
    """Return the Step 8 baseline classifier lineup."""
    return [
        ModelSpec(
            name="Dummy Stratified",
            estimator=DummyClassifier(strategy="stratified", random_state=SEED),
            notes="Chance-level baseline; every real model should beat it.",
        ),
        ModelSpec(
            name="Logistic Regression",
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            ),
            notes="Linear baseline with balanced class weights.",
        ),
        ModelSpec(
            name="K-Nearest Neighbors",
            estimator=KNeighborsClassifier(),
            notes="Distance-based non-parametric classifier.",
        ),
        ModelSpec(
            name="Support Vector Machine",
            estimator=SVC(probability=True, random_state=SEED),
            notes="Margin-based classifier with probability estimates enabled.",
        ),
        ModelSpec(
            name="Decision Tree",
            estimator=DecisionTreeClassifier(random_state=SEED),
            notes="Interpretable tree baseline.",
        ),
        ModelSpec(
            name="Random Forest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=1,
            ),
            notes="Bagged tree ensemble with balanced class weights.",
        ),
        ModelSpec(
            name="Gaussian Naive Bayes",
            estimator=GaussianNB(),
            notes=(
                "Included for lab coverage; GaussianNB assumes continuous features, "
                "which is imperfect after one-hot encoding."
            ),
        ),
        ModelSpec(
            name="Gradient Boosting",
            estimator=GradientBoostingClassifier(random_state=SEED),
            notes="Scikit-learn boosted tree baseline.",
        ),
        ModelSpec(
            name="XGBoost",
            estimator=SklearnCompatibleXGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=1,
            ),
            notes="Industry-standard boosted tree classifier for tabular data.",
        ),
    ]


def build_model_inventory(model_specs: list[ModelSpec]) -> pd.DataFrame:
    """Return a compact model inventory table."""
    return pd.DataFrame(
        [
            {
                "model": spec.name,
                "estimator": spec.estimator.__class__.__name__,
                "notes": spec.notes,
            }
            for spec in model_specs
        ]
    )


def build_model_pipeline(spec: ModelSpec) -> Pipeline:
    """Wrap a classifier in the shared preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("clf", spec.estimator),
        ]
    )


def _fold_results_from_cv(model_name: str, cv_results: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    n_folds = len(cv_results["test_accuracy"])
    for fold_index in range(n_folds):
        row = {
            "model": model_name,
            "fold": fold_index + 1,
            "fit_time_seconds": round(float(cv_results["fit_time"][fold_index]), 4),
            "score_time_seconds": round(float(cv_results["score_time"][fold_index]), 4),
        }
        for metric in SCORING:
            row[metric] = round(float(cv_results[f"test_{metric}"][fold_index]), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_fold_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize cross-validation scores as mean and standard deviation."""
    rows = []
    for model_name, group in fold_results.groupby("model", sort=False):
        row = {"model": model_name}
        for metric in SCORING:
            row[f"{metric}_mean"] = round(float(group[metric].mean()), 6)
            row[f"{metric}_std"] = round(float(group[metric].std(ddof=1)), 6)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["recall_mean", "roc_auc_mean", "f1_mean"], ascending=False)
        .reset_index(drop=True)
    )


def evaluate_baseline_models() -> BaselineModelResult:
    """Evaluate every baseline model with stratified 5-fold CV on training data only."""
    raw = load_raw_data()
    cleaned = clean_heart_data(raw).cleaned
    feature_target_split = build_feature_target_split(cleaned)
    model_specs = build_model_specs()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_tables = []
    for spec in model_specs:
        pipeline = build_model_pipeline(spec)
        cv_results = cross_validate(
            pipeline,
            feature_target_split.x_train,
            feature_target_split.y_train,
            scoring=SCORING,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        fold_tables.append(_fold_results_from_cv(spec.name, cv_results))

    fold_results = pd.concat(fold_tables, ignore_index=True)
    summary = summarize_fold_results(fold_results)

    return BaselineModelResult(
        model_inventory=build_model_inventory(model_specs),
        fold_results=fold_results,
        summary=summary,
        split_summary=feature_target_split.split.summary,
    )


def render_baseline_model_report(result: BaselineModelResult) -> str:
    """Render the Step 8 baseline model comparison report."""
    top_recall = result.summary.iloc[0]
    dummy = result.summary.loc[result.summary["model"].eq("Dummy Stratified")].iloc[0]
    real_models = result.summary.loc[~result.summary["model"].eq("Dummy Stratified")]
    best_real = real_models.iloc[0]

    return f"""# Baseline Model Comparison

## Scope

This report compares the baseline classifier lineup using stratified 5-fold cross-validation on the training split only. The held-out test set is not evaluated in this step.

## Split Summary

{result.split_summary.to_markdown(index=False)}

## Model Inventory

{result.model_inventory.to_markdown(index=False)}

## Cross-Validation Summary

Models are sorted by recall mean, then ROC-AUC mean, then F1 mean. Recall is prioritized because false negatives are costly in a heart disease screening context.

{result.summary.to_markdown(index=False)}

## Per-Fold Scores

{result.fold_results.to_markdown(index=False)}

## Initial Observations

- Top model by recall: {top_recall["model"]} with recall {top_recall["recall_mean"]:.3f} +/- {top_recall["recall_std"]:.3f}.
- Best non-dummy model by the project ranking: {best_real["model"]}.
- Dummy baseline recall: {dummy["recall_mean"]:.3f}; real models must be interpreted relative to this baseline, not accuracy alone.
- Gaussian Naive Bayes is included for coverage, but its assumptions are imperfect after one-hot encoding.
- Hyperparameter tuning should focus on the strongest 2-3 models from this baseline ranking.
"""


def write_baseline_model_outputs(result: BaselineModelResult) -> None:
    """Persist baseline model comparison tables and report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    result.model_inventory.to_csv(RESULTS_DIR / "baseline_model_inventory.csv", index=False)
    result.fold_results.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    result.summary.to_csv(RESULTS_DIR / "baseline_model_comparison.csv", index=False)
    BASELINE_MODEL_REPORT_PATH.write_text(render_baseline_model_report(result), encoding="utf-8")


def run_baseline_model_comparison() -> BaselineModelResult:
    """Run Step 8 baseline model comparison and persist outputs."""
    result = evaluate_baseline_models()
    write_baseline_model_outputs(result)
    return result


if __name__ == "__main__":
    run_baseline_model_comparison()
