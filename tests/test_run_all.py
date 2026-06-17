"""Tests for the end-to-end workflow runner."""

from src.run_all import WorkflowStep, build_execution_order_table, run_workflow_steps


def test_execution_order_table_matches_workflow_steps() -> None:
    table = build_execution_order_table()

    assert table["order"].is_monotonic_increasing
    assert table["step"].iloc[0] == "initial_data_audit"
    assert table["step"].iloc[-1] == "submission_acceptance_check"
    assert table.shape[0] == 14


def test_run_workflow_steps_executes_in_order() -> None:
    calls: list[str] = []

    def first_step() -> None:
        calls.append("first")

    def second_step() -> None:
        calls.append("second")

    steps = [
        WorkflowStep(1, "first", "First test step.", first_step),
        WorkflowStep(2, "second", "Second test step.", second_step),
    ]

    summary = run_workflow_steps(steps)

    assert calls == ["first", "second"]
    assert summary["status"].tolist() == ["passed", "passed"]
    assert summary["elapsed_seconds"].ge(0).all()
