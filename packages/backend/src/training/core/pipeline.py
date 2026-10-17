"""Training pipelines for supervised, semi-supervised, and unsupervised jobs."""

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

from src.db.models import Job
from src.training.catalogue import RL_TASK_TYPE
from src.training.charts import (
    _charts_anomaly_detection,
    _charts_classification,
    _charts_clustering,
    _charts_regression,
    _charts_semi_supervised,
    _classification_metrics,
)
from src.training.core.estimator import make_estimator
from src.training.core.preprocessor import (
    build_preprocessor,
    labelled_mask,
    semi_supervised_problem,
    semi_supervised_stats,
)


def train_pipeline(
    job: Job,
    df: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float | int], dict[str, Any]]:
    """Fit the full preprocessing + estimator pipeline and return metrics and charts."""
    feature_columns = job.feature_columns
    task_type = job.task_type
    split_ratio = job.split_ratio if job.split_ratio else 0.2
    seed = job.random_seed if job.random_seed else 42

    estimator = make_estimator(job.algorithm_id, job.hyperparameters, seed=seed)
    preprocessor = build_preprocessor(df, feature_columns)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])

    X = df[feature_columns]

    if task_type in ('classification', 'regression'):
        return _train_supervised(job, df, pipeline, X, split_ratio, seed)
    if task_type == 'semi_supervised':
        return _train_semi_supervised(job, df, pipeline, X, split_ratio, seed)
    if task_type == 'clustering':
        return _train_clustering(pipeline, X, seed)
    if task_type == 'anomaly_detection':
        return _train_anomaly_detection(pipeline, X, seed)
    if task_type == RL_TASK_TYPE:
        from src.training.rl import train_reinforcement_learning

        return train_reinforcement_learning(job, df, pipeline, seed, split_ratio)

    raise ValueError(f'INVALID_TASK_TYPE: unsupported taskType: {task_type}')


def _train_supervised(
    job: Job,
    df: pd.DataFrame,
    pipeline: Pipeline,
    X: pd.DataFrame,
    split_ratio: float,
    seed: int,
) -> tuple[Pipeline, dict[str, float | int], dict[str, Any]]:
    y = df[job.label_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_ratio,
        random_state=seed,
        stratify=y if job.task_type == 'classification' else None,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if job.task_type == 'classification':
        metrics = _classification_metrics(y_test, y_pred)
        classes = sorted(y_test.unique().astype(str))
        charts = _charts_classification(
            pipeline, job.feature_columns, X_test, y_test, y_pred, classes, seed
        )
        return pipeline, metrics, charts

    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred)),
    }
    charts = _charts_regression(pipeline, job.feature_columns, X_test, y_test.values, y_pred, seed)
    return pipeline, metrics, charts


def _train_semi_supervised(
    job: Job,
    df: pd.DataFrame,
    pipeline: Pipeline,
    X: pd.DataFrame,
    split_ratio: float,
    seed: int,
) -> tuple[Pipeline, dict[str, float | int], dict[str, Any]]:
    y_raw = df[job.label_column]
    stats = semi_supervised_stats(y_raw)
    problem = semi_supervised_problem(stats)
    if problem is not None:
        raise ValueError(f'INSUFFICIENT_LABELS: {problem}')

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
    metrics['labelledRows'] = stats['labelledRows']
    metrics['unlabelledRows'] = stats['unlabelledRows']
    metrics['iterations'] = int(getattr(pipeline.named_steps['estimator'], 'n_iter_', 0))
    estimator = pipeline.named_steps['estimator']
    classes = sorted(y_test.unique().astype(str))
    charts = _charts_semi_supervised(
        pipeline,
        job.feature_columns,
        estimator,
        classes,
        X_test,
        y_test,
        y_pred,
        stats['labelledRows'],
        stats['unlabelledRows'],
        seed,
    )
    return pipeline, metrics, charts


def _preprocessed_matrix(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.named_steps['preprocessor'].transform(X)


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
        metrics['silhouette'] = float(silhouette_score(matrix, labels))
        metrics['daviesBouldin'] = float(davies_bouldin_score(matrix, labels))
        metrics['calinskiHarabasz'] = float(calinski_harabasz_score(matrix, labels))

    estimator = pipeline.named_steps['estimator']
    if hasattr(estimator, 'inertia_'):
        metrics['inertia'] = float(estimator.inertia_)
    if hasattr(estimator, 'bic'):
        metrics['bic'] = float(estimator.bic(matrix))

    charts = _charts_clustering(pipeline, X, labels, seed)
    return pipeline, metrics, charts


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
        'anomalyCount': count,
        'anomalyRate': float(count / len(labels)),
        'scoreMean': float(np.mean(scores)),
        'scoreStd': float(np.std(scores)),
    }
    charts = _charts_anomaly_detection(scores, labels)
    return pipeline, metrics, charts
