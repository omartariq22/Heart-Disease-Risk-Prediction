"""End-to-end project execution runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import subprocess
import sys
import time
from typing import Callable

import pandas as pd

from src.data import run_initial_audit
from src.eda import run_eda
from src.evaluate import run_model_evaluation
from src.finalize import run_finalization
from src.interpret import run_feature_interpretation
from src.model_preprocess import run_preprocessing_inspection
from src.models import run_baseline_model_comparison
from src.outliers import run_outlier_detection
from src.plots import run_visualization_qa
from src.preprocess import run_data_cleaning
from src.reporting import run_reporting
from src.schema import RESULTS_DIR, run_schema_validation
from src.tune import run_hyperparameter_tuning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPRODUCIBILITY_REPORT_PATH = REPORTS_DIR / "REPRODUCIBILITY_CHECK_REPORT.md"


@dataclass(frozen=True)
class WorkflowStep:
    """Single reproducible workflow step."""

    order: int
    name: str
    description: str
    callable: Callable[[], object]


@dataclass(frozen=True)
class WorkflowResult:
    """End-to-end workflow execution result."""

    execution_summary: pd.DataFrame
    tests_ran: bool
    tests_passed: bool | None
    report_path: Path


WORKFLOW_STEPS = [
    WorkflowStep(
        1,
        "initial_data_audit",
        "Load the raw dataset and write the initial data-quality audit.",
        run_initial_audit,
    ),
    WorkflowStep(
        2,
        "schema_validation",
        "Generate the data dictionary and encoded-value validation outputs.",
        run_schema_validation,
    ),
    WorkflowStep(
        3,
        "data_cleaning",
        "Drop duplicates, decode sentinel missing values, and persist cleaning artifacts.",
        run_data_cleaning,
    ),
    WorkflowStep(
        4,
        "exploratory_data_analysis",
        "Run training-only EDA, statistical tests, and EDA figures.",
        run_eda,
    ),
    WorkflowStep(
        5,
        "outlier_detection",
        "Detect and document plausible clinical outliers on the training split.",
        run_outlier_detection,
    ),
    WorkflowStep(
        6,
        "preprocessing_inspection",
        "Inspect the shared leak-proof ColumnTransformer preprocessing pipeline.",
        run_preprocessing_inspection,
    ),
    WorkflowStep(
        7,
        "baseline_model_comparison",
        "Evaluate candidate classifiers with stratified 5-fold cross-validation.",
        run_baseline_model_comparison,
    ),
    WorkflowStep(
        8,
        "model_evaluation",
        "Generate out-of-fold evaluation metrics, curves, threshold sweeps, and confusion matrices.",
        run_model_evaluation,
    ),
    WorkflowStep(
        9,
        "hyperparameter_tuning",
        "Tune the strongest candidates and lock the selected model family.",
        run_hyperparameter_tuning,
    ),
    WorkflowStep(
        10,
        "feature_interpretation",
        "Generate coefficient, feature-importance, and permutation-importance outputs.",
        run_feature_interpretation,
    ),
    WorkflowStep(
        11,
        "visualization_qa",
        "Build the visualization manifest and required-figure coverage report.",
        run_visualization_qa,
    ),
    WorkflowStep(
        12,
        "final_model_package",
        "Fit the locked tuned model, evaluate the held-out test set once, and write the model card.",
        run_finalization,
    ),
    WorkflowStep(
        13,
        "consolidated_reporting",
        "Regenerate milestone and final project reports after final model packaging.",
        run_reporting,
    ),
]


def build_execution_order_table(steps: list[WorkflowStep] = WORKFLOW_STEPS) -> pd.DataFrame:
    """Return the documented final execution order."""
    return pd.DataFrame(
        [
            {
                "order": step.order,
                "step": step.name,
                "description": step.description,
            }
            for step in steps
        ]
    )


def run_workflow_steps(steps: list[WorkflowStep] = WORKFLOW_STEPS) -> pd.DataFrame:
    """Run workflow steps in order and capture timing/status metadata."""
    rows = []
    for step in steps:
        started = time.perf_counter()
        status = "passed"
        error_message = ""
        try:
            step.callable()
        except Exception as exc:
            status = "failed"
            error_message = f"{exc.__class__.__name__}: {exc}"
            raise
        finally:
            elapsed_seconds = time.perf_counter() - started
            rows.append(
                {
                    "order": step.order,
                    "step": step.name,
                    "status": status,
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "error_message": error_message,
                }
            )
    return pd.DataFrame(rows)


def run_pytest() -> bool:
    """Run the project test suite as part of the reproducibility check."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return completed.returncode == 0


def render_reproducibility_report(result: WorkflowResult) -> str:
    """Render a concise reproducibility check report."""
    total_seconds = result.execution_summary["elapsed_seconds"].sum()
    test_status = "not run"
    if result.tests_ran:
        test_status = "passed" if result.tests_passed else "failed"

    return f"""# Reproducibility Check Report

## Scope

This report records the final end-to-end execution order for the Heart Disease Risk Prediction project. The workflow regenerates all audit tables, cleaning outputs, EDA and outlier artifacts, model-selection outputs, final test metrics, reports, and the local model artifact from the raw dataset.

## Execution Summary

- Workflow steps executed: {len(result.execution_summary)}
- Workflow status: {"passed" if result.execution_summary["status"].eq("passed").all() else "failed"}
- Total measured workflow time: {total_seconds:.3f} seconds
- Test suite status: {test_status}

{result.execution_summary.to_markdown(index=False)}

## Notes

- Run command: `python -m src.run_all`
- Use `python -m src.run_all --skip-tests` to regenerate outputs without running `pytest`.
- The final serialized model is generated at `outputs/models/final_model.joblib` and is intentionally ignored by Git.
- The reproducibility contract uses the fixed global seed `src.SEED = 42`.
"""


def write_workflow_outputs(result: WorkflowResult) -> None:
    """Persist reproducibility tables and report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    build_execution_order_table().to_csv(
        RESULTS_DIR / "reproducibility_execution_order.csv",
        index=False,
    )
    result.execution_summary.to_csv(
        RESULTS_DIR / "reproducibility_run_summary.csv",
        index=False,
    )
    result.report_path.write_text(render_reproducibility_report(result), encoding="utf-8")


def run_all(skip_tests: bool = False) -> WorkflowResult:
    """Run the full project workflow and optionally the test suite."""
    execution_summary = run_workflow_steps()
    tests_passed = None
    if not skip_tests:
        tests_passed = run_pytest()
        if not tests_passed:
            raise RuntimeError("End-to-end workflow completed, but pytest failed.")

    result = WorkflowResult(
        execution_summary=execution_summary,
        tests_ran=not skip_tests,
        tests_passed=tests_passed,
        report_path=REPRODUCIBILITY_REPORT_PATH,
    )
    write_workflow_outputs(result)
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the workflow runner."""
    parser = argparse.ArgumentParser(description="Run the full heart disease project workflow.")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Regenerate project outputs without running pytest at the end.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(skip_tests=args.skip_tests)
