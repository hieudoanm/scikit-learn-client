"""Unit tests for src/skl/cli.py."""

from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

from skl.cli import cli


class TestCliRun:
    def test_run_regression_flags(self, tmp_path):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}).to_csv(
            csv, index=False
        )
        out = tmp_path / "model.onnx"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(csv),
                "--features",
                "x",
                "--label",
                "y",
                "--task-type",
                "regression",
                "--algorithm",
                "linear_regression",
                "--format",
                "onnx",
                "--output",
                str(out),
                "--no-interactive",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_run_default_output(self, tmp_path):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 4.0]}).to_csv(csv, index=False)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(csv),
                "--features",
                "x",
                "--label",
                "y",
                "--task-type",
                "regression",
                "--algorithm",
                "linear_regression",
                "--no-interactive",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (csv.parent / "data.onnx").exists()

    def test_run_missing_required_flag_no_interactive(self, tmp_path):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"x": [1.0], "y": [2.0]}).to_csv(csv, index=False)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(csv),
                "--no-interactive",
            ],
        )
        assert result.exit_code != 0

    def test_run_missing_features_no_interactive(self, tmp_path):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"x": [1.0], "y": [2.0]}).to_csv(csv, index=False)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(csv),
                "--task-type",
                "regression",
                "--algorithm",
                "linear_regression",
                "--no-interactive",
            ],
        )
        assert result.exit_code != 0

    def test_run_clustering_no_label(self, tmp_path):
        csv = tmp_path / "data.csv"
        pd.DataFrame({"x1": [0.0, 0.1, 10.0], "x2": [0.0, 0.1, 10.0]}).to_csv(csv, index=False)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(csv),
                "--features",
                "x1,x2",
                "--task-type",
                "clustering",
                "--algorithm",
                "kmeans",
                "--format",
                "onnx",
                "--no-interactive",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
