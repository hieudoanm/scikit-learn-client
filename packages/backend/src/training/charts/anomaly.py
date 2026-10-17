"""Anomaly-detection chart generation for training results."""

from typing import Any

import numpy as np

MAX_POINTS = 500
MAX_BINS = 50


def _charts_anomaly_detection(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Generate chart data for anomaly detection tasks."""
    # Score histogram
    hist, edges = np.histogram(scores, bins=MAX_BINS)
    # Threshold: score at which a row becomes an anomaly
    anomaly_mask = labels == -1
    threshold = None
    if anomaly_mask.any():
        threshold = float(np.max(scores[anomaly_mask]))
    # Sample points for score-by-order plot
    n = len(scores)
    if n > MAX_POINTS:
        idx = np.linspace(0, n - 1, MAX_POINTS, dtype=int)
        scores_s = scores[idx]
        labels_s = labels[idx]
        indices_s = np.arange(n)[idx]
    else:
        scores_s = scores
        labels_s = labels
        indices_s = np.arange(n)
    return {
        'scoreHistogram': {
            'edges': edges.tolist(),
            'counts': hist.tolist(),
            'threshold': threshold,
        },
        'scorePoints': {
            'index': indices_s.tolist(),
            'score': scores_s.tolist(),
            'anomaly': (labels_s == -1).tolist(),
        },
    }
