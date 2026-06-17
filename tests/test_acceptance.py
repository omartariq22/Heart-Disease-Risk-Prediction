"""Tests for the final submission acceptance checklist."""

from src.acceptance import (
    build_artifact_checklist,
    build_success_criteria_checklist,
    run_acceptance_check,
)


def test_acceptance_checklists_pass_for_generated_project_artifacts() -> None:
    artifacts = build_artifact_checklist()
    criteria = build_success_criteria_checklist()

    assert artifacts["exists"].all()
    assert criteria["passed"].all()


def test_run_acceptance_check_writes_summary_report() -> None:
    result = run_acceptance_check()

    assert result.report_path.exists()
    assert result.summary["metric"].tolist()[:2] == [
        "required_artifacts_present",
        "success_criteria_passed",
    ]
    assert "Submission readiness: **passed**" in result.report_path.read_text(encoding="utf-8")
