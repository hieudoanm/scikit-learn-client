"""Clustering chart generation for training results."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

MAX_POINTS = 500


def _charts_clustering(
    pipeline: Pipeline,
    X: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Generate chart data for clustering tasks."""
    matrix = pipeline.named_steps['preprocessor'].transform(X)
    n_clusters = len(np.unique(labels))
    # Cluster sizes
    cluster_sizes = [int((labels == i).sum()) for i in range(n_clusters)]
    # PCA projection for scatter (seeded sample up to MAX_POINTS)
    rng = np.random.default_rng(seed)
    n = len(matrix)
    if n > MAX_POINTS:
        idx = rng.choice(n, MAX_POINTS, replace=False)
        matrix_s = matrix[idx]
        labels_s = labels[idx]
    else:
        matrix_s = matrix
        labels_s = labels
    # Fit PCA on the full matrix for consistency, then transform sample
    pca = PCA(n_components=2, random_state=seed)
    proj = pca.fit_transform(matrix_s)
    projection = {
        'x': proj[:, 0].tolist(),
        'y': proj[:, 1].tolist(),
        'cluster': labels_s.tolist(),
    }
    # Silhouette samples (if silhouette was computed)
    silhouette_samples = None
    if 1 < n_clusters < len(matrix):
        try:
            from sklearn.metrics import silhouette_samples as sk_silhouette_samples

            sil = sk_silhouette_samples(matrix, labels)
            if len(sil) > MAX_POINTS:
                idx = rng.choice(len(sil), MAX_POINTS, replace=False)
                sil = sil[idx]
                labels_sil = labels[idx]
            else:
                labels_sil = labels
            silhouette_samples = {
                'value': sil.tolist(),
                'cluster': labels_sil.tolist(),
            }
        except Exception:  # noqa: BLE001,S110
            # Silhouette samples may fail for degenerate cases (e.g., all rows identical)
            pass
    charts: dict[str, Any] = {
        'clusterSizes': cluster_sizes,
        'projection': projection,
    }
    if silhouette_samples is not None:
        charts['silhouetteSamples'] = silhouette_samples
    return charts
