"""Regression chart generation for training results."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.training.charts.feature_importance import _collect_feature_importance
from src.training.charts.mlp import _mlp_progress

MAX_POINTS = 500


def _charts_regression(
    pipeline: Pipeline,
    feature_columns: list[str],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Generate chart data for regression tasks."""
    rng = np.random.default_rng(seed)
    n = len(y_test)
    # Sample up to MAX_POINTS
    if n > MAX_POINTS:
        idx = rng.choice(n, MAX_POINTS, replace=False)
        y_test_s = y_test[idx]
        y_pred_s = y_pred[idx]
    else:
        y_test_s = y_test
        y_pred_s = y_pred
    # Residuals
    residuals = y_test_s - y_pred_s

    # Feature importance
    feature_importance = _collect_feature_importance(
        pipeline, feature_columns, X_test, y_test, 'regression', seed
    )
    result = {
        'regressionPoints': {
            'predicted': y_pred_s.tolist(),
            'actual': y_test_s.tolist(),
        },
        'residuals': {
            'predicted': y_pred_s.tolist(),
            'residuals': residuals.tolist(),
        },
    }
    if feature_importance:
        result['featureImportance'] = feature_importance
    mlp_progress = _mlp_progress(pipeline)
    if mlp_progress:
        result['mlpProgress'] = mlp_progress
    return result
