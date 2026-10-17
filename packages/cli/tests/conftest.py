"""Shared fixtures for the skl CLI test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def regression_csv(tmp_path: Path) -> Path:
    """A small CSV where y = 2x (perfect linear regression)."""
    p = tmp_path / "regression.csv"
    pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]}
    ).to_csv(p, index=False)
    return p


@pytest.fixture
def classification_csv(tmp_path: Path) -> Path:
    """A small labelled classification CSV."""
    p = tmp_path / "classification.csv"
    pd.DataFrame(
        {
            "a": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "label": ["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"],
        }
    ).to_csv(p, index=False)
    return p


@pytest.fixture
def clustering_csv(tmp_path: Path) -> Path:
    """A small unlabelled clustering CSV."""
    p = tmp_path / "clustering.csv"
    pd.DataFrame({"x1": [0.0, 0.1, 10.0, 10.1], "x2": [0.0, 0.1, 10.0, 10.1]}).to_csv(
        p, index=False
    )
    return p


@pytest.fixture
def mixed_csv(tmp_path: Path) -> Path:
    """A CSV with numeric and categorical columns."""
    p = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
            "label": [0, 1, 0],
        }
    ).to_csv(p, index=False)
    return p


@pytest.fixture
def columns() -> list[dict]:
    return [{"name": "x"}, {"name": "y"}, {"name": "label"}]
