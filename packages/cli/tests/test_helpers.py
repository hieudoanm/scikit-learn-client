"""Unit tests for src/skl/helpers.py."""

from __future__ import annotations

import pytest

import skl.helpers as h


class TestPrintHelpers:
    def test_print_info(self, capsys):
        h.print_info("hello")
        assert "hello" in capsys.readouterr().out

    def test_print_success(self, capsys):
        h.print_success("ok")
        assert "ok" in capsys.readouterr().out

    def test_print_warning(self, capsys):
        h.print_warning("warn")
        assert "warn" in capsys.readouterr().out

    def test_print_error(self, capsys):
        h.print_error("err")
        captured = capsys.readouterr()
        assert "err" in captured.err

    def test_format_size(self):
        assert h.format_size(0) == "0.0 B"
        assert h.format_size(500) == "500.0 B"
        assert h.format_size(1500).endswith("KB")
        assert h.format_size(1_500_000).endswith("MB")
        assert h.format_size(1_500_000_000).endswith("GB")
        assert h.format_size(10**15).endswith("TB")


class TestChooseHelpers:
    def test_choose_task_type(self, monkeypatch, columns):
        monkeypatch.setattr(
            h.inquirer,
            "prompt",
            lambda questions: {"task_type": "regression — Regression — predict a continuous value"},
        )
        assert h.choose_task_type(columns) == "regression"

    def test_choose_task_type_no_answer_exits(self, monkeypatch, columns):
        monkeypatch.setattr(h.inquirer, "prompt", lambda questions: None)
        with pytest.raises(SystemExit):
            h.choose_task_type(columns)

    def test_choose_columns_multi(self, monkeypatch, columns):
        monkeypatch.setattr(h.inquirer, "prompt", lambda questions: {"columns": ["x", "y"]})
        assert h.choose_columns(columns, "pick") == ["x", "y"]

    def test_choose_columns_single(self, monkeypatch, columns):
        monkeypatch.setattr(h.inquirer, "prompt", lambda questions: {"column": "x"})
        assert h.choose_columns(columns, "pick", multi=False) == "x"

    def test_choose_columns_no_answer_exits(self, monkeypatch, columns):
        monkeypatch.setattr(h.inquirer, "prompt", lambda questions: None)
        with pytest.raises(SystemExit):
            h.choose_columns(columns, "pick")

    def test_choose_algorithm(self, monkeypatch):
        algos = [
            {
                "name": "Logistic Regression",
                "algorithmId": "logistic_regression",
                "task_type": "classification",
            },
            {
                "name": "Random Forest",
                "algorithmId": "random_forest_classifier",
                "task_type": "classification",
            },
        ]
        monkeypatch.setattr(
            h.inquirer,
            "prompt",
            lambda questions: {"algorithm": "Random Forest (random_forest_classifier)"},
        )
        result = h.choose_algorithm(algos, "classification")
        assert result["algorithmId"] == "random_forest_classifier"

    def test_choose_algorithm_no_answer_exits(self, monkeypatch):
        monkeypatch.setattr(h.inquirer, "prompt", lambda questions: None)
        with pytest.raises(SystemExit):
            h.choose_algorithm([], "classification")

    def test_confirm_export_true(self, monkeypatch):
        monkeypatch.setattr(h.inquirer, "prompt", lambda questions: {"export": True})
        assert h.confirm_export([{"format": "onnx", "filename": "m.onnx"}]) is True

    def test_confirm_export_false(self, monkeypatch):
        monkeypatch.setattr(h.inquirer, "prompt", lambda questions: {"export": False})
        assert h.confirm_export([{"format": "onnx", "filename": "m.onnx"}]) is False

    def test_confirm_export_no_answer(self, monkeypatch):
        monkeypatch.setattr(h.inquirer, "prompt", lambda questions: None)
        assert h.confirm_export([{"format": "onnx", "filename": "m.onnx"}]) is False


class TestDisplayHelpers:
    def test_display_job(self):
        job = {
            "status": "completed",
            "taskType": "regression",
            "algorithmId": "linear_regression",
            "featureColumns": ["x"],
            "labelColumn": "y",
            "metrics": {"r2": 1.0},
            "artefacts": [{"format": "onnx", "filename": "m.onnx", "byteSize": 100}],
        }
        h.display_job(job)

    def test_display_job_no_label(self):
        job = {
            "status": "completed",
            "taskType": "clustering",
            "algorithmId": "kmeans",
            "featureColumns": ["x"],
            "metrics": {},
            "artefacts": [],
        }
        h.display_job(job)

    def test_display_job_with_error(self):
        job = {
            "status": "failed",
            "taskType": "regression",
            "algorithmId": "a",
            "featureColumns": [],
            "error": "boom",
        }
        h.display_job(job)

    def test_display_jobs_empty(self):
        h.display_jobs([])

    def test_display_jobs_list(self):
        jobs = [
            {
                "jobId": "abc12345",
                "status": "completed",
                "taskType": "regression",
                "algorithmId": "linear_regression",
                "name": None,
                "createdAt": "2024-01-01",
            }
        ]
        h.display_jobs(jobs)
