"""Unit tests for src/skl/catalogue modules."""

from __future__ import annotations

from skl.catalogue import ALGORITHMS, LABEL_REQUIRED_TASKS, TASK_TYPES


class TestCatalogue:
    def test_algorithms_non_empty(self):
        assert ALGORITHMS

    def test_each_algorithm_has_required_keys(self):
        for _aid, entry in ALGORITHMS.items():
            assert "name" in entry
            assert "task_type" in entry
            assert "default_hyperparameters" in entry
            assert "exports" in entry
            assert "estimator" in entry
            assert entry["task_type"] in TASK_TYPES

    def test_label_required_tasks(self):
        assert "classification" in LABEL_REQUIRED_TASKS
        assert "regression" in LABEL_REQUIRED_TASKS
        assert "semi_supervised" in LABEL_REQUIRED_TASKS
        assert "clustering" not in LABEL_REQUIRED_TASKS

    def test_task_types(self):
        assert "classification" in TASK_TYPES
        assert "regression" in TASK_TYPES
        assert "clustering" in TASK_TYPES
