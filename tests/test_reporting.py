"""Tests for consolidated report generation."""

from src.reporting import run_reporting


def test_reporting_generates_milestone_and_final_reports(tmp_path) -> None:
    result = run_reporting(reports_dir=tmp_path)

    milestone = result.milestone_report_path.read_text(encoding="utf-8")
    final = result.final_report_path.read_text(encoding="utf-8")

    assert result.milestone_report_path.exists()
    assert result.final_report_path.exists()
    assert "Milestone 1 Report" in milestone
    assert "Data Cleaning Summary" in milestone
    assert "Visualization Coverage" in milestone
    assert "Final Project Report" in final
    assert "Classification Models" in final
    assert "Support Vector Machine" in final
    assert "Final Held-Out Test Evaluation" in final
