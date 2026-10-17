"""Training pipelines for supervised, semi-supervised, and unsupervised jobs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .estimator import make_estimator
from .preprocessor import (
    build_preprocessor,
    labelled_mask,
    semi_supervised_problem,
    semi_supervised_stats,
)


def _classification_metrics(y_test: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precisionMacro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recallMacro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1Macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def train_pipeline(
    feature_columns: list[str],
    label_column: str | None,
    task_type: str,
    algorithm_id: str,
    hyperparameters: dict | None,
    df: pd.DataFrame,
    split_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, float | int], dict[str, Any]]:
    """Fit the full preprocessing + estimator pipeline and return metrics and charts."""
    estimator = make_estimator(algorithm_id, hyperparameters, seed=seed)
    preprocessor = build_preprocessor(df, feature_columns)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])

    X = df[feature_columns]

    if task_type in ("classification", "regression"):
        return _train_supervised(
            feature_columns, label_column, task_type, df, pipeline, X, split_ratio, seed
        )
    if task_type == "semi_supervised":
        return _train_semi_supervised(
            feature_columns, label_column, df, pipeline, X, split_ratio, seed
        )
    if task_type == "clustering":
        return _train_clustering(pipeline, X, seed)
    if task_type == "anomaly_detection":
        return _train_anomaly_detection(pipeline, X, seed)

    raise ValueError(f"INVALID_TASK_TYPE: unsupported taskType: {task_type}")


def _train_supervised(
    feature_columns: list[str],
    label_column: str | None,
    task_type: str,
    df: pd.DataFrame,
    pipeline: Pipeline,
    X: pd.DataFrame,
    split_ratio: float,
    seed: int,
) -> tuple[Pipeline, dict[str, float | int], dict[str, Any]]:
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_ratio,
        random_state=seed,
        stratify=y if task_type == "classification" else None,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if task_type == "classification":
        metrics = _classification_metrics(y_test, y_pred)
        return pipeline, metrics, {}

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    return pipeline, metrics, {}


def _train_semi_supervised(
    feature_columns: list[str],
    label_column: str | None,
    df: pd.DataFrame,
    pipeline: Pipeline,
    X: pd.DataFrame,
    split_ratio: float,
    seed: int,
) -> tuple[Pipeline, dict[str, float | int], dict[str, Any]]:
    y_raw = df[label_column]
    stats = semi_supervised_stats(y_raw)
    problem = semi_supervised_problem(stats)
    if problem is not None:
        raise ValueError(f"INSUFFICIENT_LABELS: {problem}")

    mask = labelled_mask(y_raw)
    X_labelled = X[mask]
    y_labelled = y_raw[mask]
    X_unlabelled = X[~mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_labelled,
        y_labelled,
        test_size=split_ratio,
        random_state=seed,
        stratify=y_labelled,
    )

    if len(X_unlabelled) > 0:
        X_train = pd.concat([X_train, X_unlabelled], axis=0)
        y_train = pd.concat(
            [y_train.reset_index(drop=True), pd.Series([-1] * len(X_unlabelled))],
            ignore_index=True,
        )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics: dict[str, float | int] = _classification_metrics(y_test, y_pred)
    metrics["labelledRows"] = stats["labelledRows"]
    metrics["unlabelledRows"] = stats["unlabelledRows"]
    n_iter = getattr(pipeline.named_steps["estimator"], "n_iter_", 0)
    if hasattr(n_iter, "__len__") and not isinstance(n_iter, (str, bytes)):
        n_iter = int(n_iter[0]) if len(n_iter) > 0 else 0
    metrics["iterations"] = int(n_iter)
    return pipeline, metrics, {}


def _preprocessed_matrix(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.named_steps["preprocessor"].transform(X)


def _train_clustering(
    pipeline: Pipeline,
    X: pd.DataFrame,
    seed: int,
) -> tuple[Pipeline, dict[str, float], dict[str, Any]]:
    pipeline.fit(X)
    labels = pipeline.predict(X)
    matrix = _preprocessed_matrix(pipeline, X)

    metrics: dict[str, float] = {}
    unique = np.unique(labels)
    if 1 < len(unique) < len(matrix):
        metrics["silhouette"] = float(silhouette_score(matrix, labels))
        metrics["daviesBouldin"] = float(davies_bouldin_score(matrix, labels))
        metrics["calinskiHarabasz"] = float(calinski_harabasz_score(matrix, labels))

    estimator = pipeline.named_steps["estimator"]
    if hasattr(estimator, "inertia_"):
        metrics["inertia"] = float(estimator.inertia_)
    return pipeline, metrics, {}


def _train_anomaly_detection(
    pipeline: Pipeline,
    X: pd.DataFrame,
    seed: int,
) -> tuple[Pipeline, dict[str, float], dict[str, Any]]:
    pipeline.fit(X)
    labels = np.asarray(pipeline.predict(X))
    scores = np.asarray(pipeline.decision_function(X))
    count = int((labels == -1).sum())
    metrics = {
        "anomalyCount": count,
        "anomalyRate": float(count / len(labels)),
        "scoreMean": float(np.mean(scores)),
        "scoreStd": float(np.std(scores)),
    }
    return pipeline, metrics, {}
