"""Unit tests for src/skl/pipeline.py."""

from __future__ import annotations

import pandas as pd
import pytest

from skl.pipeline import train_pipeline


class TestTrainPipelineRegression:
    def test_regression_returns_metrics(self, regression_csv):
        df = pd.read_csv(regression_csv)
        pipe, metrics, _ = train_pipeline(
            feature_columns=["x"],
            label_column="y",
            task_type="regression",
            algorithm_id="linear_regression",
            hyperparameters={},
            df=df,
        )
        assert pipe is not None
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["r2"] > 0.99

    def test_decision_tree_regressor(self, regression_csv):
        df = pd.read_csv(regression_csv)
        pipe, metrics, _ = train_pipeline(
            feature_columns=["x"],
            label_column="y",
            task_type="regression",
            algorithm_id="decision_tree_regressor",
            hyperparameters={},
            df=df,
        )
        assert "r2" in metrics


class TestTrainPipelineClassification:
    def test_classification_returns_metrics(self, classification_csv):
        df = pd.read_csv(classification_csv)
        pipe, metrics, _ = train_pipeline(
            feature_columns=["a", "b"],
            label_column="label",
            task_type="classification",
            algorithm_id="logistic_regression",
            hyperparameters={},
            df=df,
        )
        assert pipe is not None
        assert "accuracy" in metrics
        assert "f1Macro" in metrics


class TestTrainPipelineClustering:
    def test_clustering_returns_metrics(self, clustering_csv):
        df = pd.read_csv(clustering_csv)
        pipe, metrics, _ = train_pipeline(
            feature_columns=["x1", "x2"],
            label_column=None,
            task_type="clustering",
            algorithm_id="kmeans",
            hyperparameters={"n_clusters": 2},
            df=df,
        )
        assert pipe is not None
        assert "silhouette" in metrics


class TestTrainPipelineAnomalyDetection:
    def test_anomaly_detection_returns_metrics(self, clustering_csv):
        df = pd.read_csv(clustering_csv)
        pipe, metrics, _ = train_pipeline(
            feature_columns=["x1", "x2"],
            label_column=None,
            task_type="anomaly_detection",
            algorithm_id="isolation_forest",
            hyperparameters={"contamination": 0.25},
            df=df,
        )
        assert pipe is not None
        assert "anomalyCount" in metrics
        assert "anomalyRate" in metrics


class TestTrainPipelineSemiSupervised:
    def test_semi_supervised_returns_metrics(self, classification_csv):
        df = pd.read_csv(classification_csv)
        # Use numeric labels (0/1) with NaN for unlabelled; labelled_mask uses notna() for numeric
        df = pd.concat([df] * 2, ignore_index=True)
        df["label"] = (df["label"] == "cat").astype(float)
        df.loc[0, "label"] = float("nan")
        df.loc[1, "label"] = float("nan")
        pipe, metrics, _ = train_pipeline(
            feature_columns=["a", "b"],
            label_column="label",
            task_type="semi_supervised",
            algorithm_id="logistic_regression",
            hyperparameters={},
            df=df,
        )
        assert pipe is not None
        assert "labelledRows" in metrics
        assert "unlabelledRows" in metrics

    def test_semi_supervised_insufficient_labels_raises(self, classification_csv):
        df = pd.read_csv(classification_csv)
        df.loc[:, "label"] = None  # all unlabelled
        with pytest.raises(ValueError, match="INSUFFICIENT_LABELS"):
            train_pipeline(
                feature_columns=["a", "b"],
                label_column="label",
                task_type="semi_supervised",
                algorithm_id="logistic_regression",
                hyperparameters={},
                df=df,
            )


class TestTrainPipelineInvalidTask:
    def test_invalid_task_type_raises(self, regression_csv):
        df = pd.read_csv(regression_csv)
        with pytest.raises(ValueError, match="INVALID_TASK_TYPE"):
            train_pipeline(
                feature_columns=["x"],
                label_column="y",
                task_type="reinforcement_learning",
                algorithm_id="linear_regression",
                hyperparameters={},
                df=df,
            )
