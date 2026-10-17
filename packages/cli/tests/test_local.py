"""Unit tests for src/skl/local.py."""

from __future__ import annotations

import pytest

from skl.local import (
    EXPORTERS,
    EXTENSIONS,
    algorithms_for_task,
    default_output_path,
    train_and_export,
)


class TestAlgorithmsForTask:
    def test_known_task_type_returns_entries(self, columns):
        result = algorithms_for_task("regression")
        assert isinstance(result, list)
        assert len(result) > 0
        for entry in result:
            assert entry["task_type"] == "regression"
            assert "algorithmId" in entry
            assert "name" in entry

    def test_unknown_task_type_raises(self):
        with pytest.raises(ValueError, match="UNKNOWN_TASK_TYPE"):
            algorithms_for_task("not_a_task_type")


class TestDefaultOutputPath:
    def test_onnx_extension(self, tmp_path):
        csv = tmp_path / "data.csv"
        assert default_output_path(csv, "onnx") == tmp_path / "data.onnx"

    def test_pkl_extension(self, tmp_path):
        csv = tmp_path / "data.csv"
        assert default_output_path(csv, "pkl") == tmp_path / "data.pkl"

    def test_joblib_extension(self, tmp_path):
        csv = tmp_path / "data.csv"
        assert default_output_path(csv, "joblib") == tmp_path / "data.joblib"

    def test_preserves_directory(self, tmp_path):
        csv = tmp_path / "sub" / "data.csv"
        assert default_output_path(csv, "onnx") == tmp_path / "sub" / "data.onnx"


class TestTrainAndExportRegression:
    def test_onnx_end_to_end(self, regression_csv, tmp_path):
        out = tmp_path / "out.onnx"
        result = train_and_export(
            csv_path=regression_csv,
            output=out,
            task_type="regression",
            feature_columns=["x"],
            label_column="y",
            algorithm_id="linear_regression",
            export_format="onnx",
        )
        assert result == out
        assert out.exists()
        assert out.read_bytes()

    def test_pkl_end_to_end(self, regression_csv, tmp_path):
        out = tmp_path / "out.pkl"
        train_and_export(
            csv_path=regression_csv,
            output=out,
            task_type="regression",
            feature_columns=["x"],
            label_column="y",
            algorithm_id="linear_regression",
            export_format="pkl",
        )
        assert out.exists()

    def test_joblib_end_to_end(self, regression_csv, tmp_path):
        out = tmp_path / "out.joblib"
        train_and_export(
            csv_path=regression_csv,
            output=out,
            task_type="regression",
            feature_columns=["x"],
            label_column="y",
            algorithm_id="linear_regression",
            export_format="joblib",
        )
        assert out.exists()

    def test_default_output_format(self, regression_csv, tmp_path):
        out = train_and_export(
            csv_path=regression_csv,
            output=tmp_path / "model.onnx",
            task_type="regression",
            feature_columns=["x"],
            label_column="y",
            algorithm_id="linear_regression",
        )
        assert out.suffix == ".onnx"

    def test_missing_columns_raises(self, regression_csv):
        with pytest.raises(ValueError, match="columns not found in CSV"):
            train_and_export(
                csv_path=regression_csv,
                output=regression_csv.parent / "out.onnx",
                task_type="regression",
                feature_columns=["x", "nonexistent"],
                label_column="y",
                algorithm_id="linear_regression",
            )

    def test_missing_label_raises(self, regression_csv):
        with pytest.raises(ValueError, match="label column required"):
            train_and_export(
                csv_path=regression_csv,
                output=regression_csv.parent / "out.onnx",
                task_type="regression",
                feature_columns=["x"],
                label_column=None,
                algorithm_id="linear_regression",
            )

    def test_invalid_algorithm_for_task_raises(self, regression_csv):
        with pytest.raises(ValueError, match="is not valid for task type"):
            train_and_export(
                csv_path=regression_csv,
                output=regression_csv.parent / "out.onnx",
                task_type="regression",
                feature_columns=["x"],
                label_column="y",
                algorithm_id="kmeans",
            )

    def test_unknown_algorithm_raises(self, regression_csv):
        with pytest.raises(ValueError, match="is not valid for task type"):
            train_and_export(
                csv_path=regression_csv,
                output=regression_csv.parent / "out.onnx",
                task_type="regression",
                feature_columns=["x"],
                label_column="y",
                algorithm_id="does_not_exist",
            )


class TestTrainAndExportClassification:
    def test_classification_end_to_end(self, classification_csv, tmp_path):
        out = tmp_path / "out.onnx"
        train_and_export(
            csv_path=classification_csv,
            output=out,
            task_type="classification",
            feature_columns=["a", "b"],
            label_column="label",
            algorithm_id="logistic_regression",
            export_format="onnx",
        )
        assert out.exists()


class TestTrainAndExportClustering:
    def test_clustering_no_label(self, clustering_csv, tmp_path):
        out = tmp_path / "out.onnx"
        train_and_export(
            csv_path=clustering_csv,
            output=out,
            task_type="clustering",
            feature_columns=["x1", "x2"],
            label_column=None,
            algorithm_id="kmeans",
            export_format="onnx",
        )
        assert out.exists()


class TestExportersAndExtensions:
    def test_exporters_map(self):
        assert EXPORTERS["onnx"].__name__ == "export_onnx"
        assert EXPORTERS["pkl"].__name__ == "export_pickle"
        assert EXPORTERS["joblib"].__name__ == "export_joblib"

    def test_extensions_map(self):
        assert EXTENSIONS == {"onnx": ".onnx", "pkl": ".pkl", "joblib": ".joblib"}
