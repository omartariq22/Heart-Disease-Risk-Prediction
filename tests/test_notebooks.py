"""Smoke tests for deliverable notebooks."""

from pathlib import Path

import nbformat


def test_deliverable_notebooks_are_valid_and_ordered() -> None:
    notebook_paths = [
        Path("notebooks/01_eda.ipynb"),
        Path("notebooks/02_modelling.ipynb"),
    ]

    for path in notebook_paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.cells[0].cell_type == "markdown"
        assert any(cell.cell_type == "code" for cell in notebook.cells)
