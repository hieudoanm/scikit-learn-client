"""Chart generation utilities for training results.

Classification and semi-supervised charts live here; the other task families
live in :mod:`charts_regression`, :mod:`charts_clustering`, and
:mod:`charts_anomaly` and are re-exported for a single import site.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import SelfTrainingClassifier

from src.training.charts.anomaly import _charts_anomaly_detection
from src.training.charts.clustering import _charts_clustering
from src.training.charts.feature_importance import _collect_feature_importance
from src.training.charts.mlp import _mlp_progress
from src.training.charts.regression import _charts_regression

__all__ = [
    'MAX_BINS',
    'MAX_CLASSES',
    'MAX_POINTS',
    '_charts_anomaly_detection',
    '_charts_classification',
    '_charts_clustering',
    '_charts_regression',
    '_charts_semi_supervised',
    '_classification_metrics',
]

MAX_POINTS = 500
MAX_CLASSES = 50
MAX_BINS = 50


def _classification_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precisionMacro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recallMacro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1Macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def _charts_classification(
    pipeline: Pipeline,
    feature_columns: list[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    classes: list[str],
    seed: int,
) -> dict[str, Any]:
    """Generate chart data for classification tasks."""
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    # Per-class metrics
    per_class = []
    for _i, cls in enumerate(classes):
        support = int((y_test == cls).sum())
        if support > 0:
            precision = float(
                precision_score(y_test, y_pred, labels=[cls], average=None, zero_division=0)[0]
            )
            recall = float(
                recall_score(y_test, y_pred, labels=[cls], average=None, zero_division=0)[0]
            )
            f1 = float(f1_score(y_test, y_pred, labels=[cls], average=None, zero_division=0)[0])
        else:
            precision = recall = f1 = 0.0
        per_class.append(
            {
                'class': cls,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
            }
        )
    # Feature importance
    feature_importance = _collect_feature_importance(
        pipeline, feature_columns, X_test, y_test.values, 'classification', seed
    )
    result = {
        'classes': classes,
        'confusionMatrix': cm.tolist(),
        'perClass': per_class,
    }
    if feature_importance:
        result['featureImportance'] = feature_importance
    mlp_progress = _mlp_progress(pipeline)
    if mlp_progress:
        result['mlpProgress'] = mlp_progress
    return result


def _charts_semi_supervised(
    pipeline: Pipeline,
    feature_columns: list[str],
    estimator: SelfTrainingClassifier,
    classes: list[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    labelled_rows: int,
    unlabelled_rows: int,
    seed: int,
) -> dict[str, Any]:
    """Generate chart data for semi-supervised tasks."""
    base = _charts_classification(pipeline, feature_columns, X_test, y_test, y_pred, classes, seed)
    # Self-training progress from labeled_iter_
    labeled_iter = getattr(estimator, 'labeled_iter_', None)
    newly_labelled = []
    if labeled_iter is not None:
        max_iter = int(np.max(labeled_iter[labeled_iter >= 0]))
        for i in range(1, max_iter + 1):
            newly_labelled.append(int(np.sum(labeled_iter == i)))
    base['selfTraining'] = {
        'iterations': int(getattr(estimator, 'n_iter_', 0)),
        'newlyLabelled': newly_labelled,
    }
    base['labelCoverage'] = {
        'labelled': labelled_rows,
        'unlabelled': unlabelled_rows,
    }
    return base
