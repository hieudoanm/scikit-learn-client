"""Contract tests for the CSV fixtures in ``tests/fixtures``.

The fixtures mirror the table in
``docs/09-testing-strategy/02-test-fixtures.md``: deterministic row order,
UTF-8, no PII, and an empty label cell (not a sentinel token) for unlabelled
rows. Each file is asserted against its documented shape and, where it maps to
a task type, driven through the real training pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.db.models import Job
from src.training.core.pipeline import train_pipeline
from src.training.core.preprocessor import (
    semi_supervised_problem,
    semi_supervised_stats,
)

FIXTURES = Path(__file__).parent / 'fixtures'

DOCUMENTED_FILES = [
    'binary_class.csv',
    'regression.csv',
    'semi_supervised.csv',
    'semi_supervised_insufficient.csv',
    'clustering.csv',
    'anomaly.csv',
    'empty.csv',
    'header_only.csv',
    'non_numeric_y.csv',
]


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name)


def _job(
    algorithm_id: str,
    task_type: str,
    label_column: str | None,
    hyperparameters: dict | None = None,
) -> Job:
    return Job(
        feature_columns=['feature1', 'feature2'],
        label_column=label_column,
        task_type=task_type,
        algorithm_id=algorithm_id,
        split_ratio=0.25,
        random_seed=42,
        hyperparameters=hyperparameters or {},
    )


def test_documented_fixtures_exist():
    """Every fixture listed in the testing strategy is present."""
    missing = [name for name in DOCUMENTED_FILES if not (FIXTURES / name).exists()]
    assert missing == []


def test_classification_fixture_contract():
    df = _load('binary_class.csv')
    assert list(df.columns) == ['feature1', 'feature2', 'label']
    assert len(df) >= 40
    assert df['label'].nunique() == 2


def test_regression_fixture_contract():
    df = _load('regression.csv')
    assert list(df.columns) == ['feature1', 'feature2', 'label']
    assert pd.api.types.is_numeric_dtype(df['label'])
    assert len(df) >= 40


def test_semi_supervised_fixture_contract():
    """Two classes, at least 40 labelled rows, and at least 10 blank cells (FR-081..FR-082)."""
    df = _load('semi_supervised.csv')
    assert list(df.columns) == ['feature1', 'feature2', 'label']

    stats = semi_supervised_stats(df['label'])
    assert stats['labelledRows'] >= 40
    assert stats['unlabelledRows'] >= 10
    assert semi_supervised_problem(stats) is None


def test_semi_supervised_insufficient_fixture_contract():
    """Fewer than 10 labelled rows must be rejected (FR-082)."""
    df = _load('semi_supervised_insufficient.csv')
    stats = semi_supervised_stats(df['label'])
    assert stats['labelledRows'] < 10
    assert semi_supervised_problem(stats) is not None


def test_clustering_fixture_contract():
    df = _load('clustering.csv')
    assert list(df.columns) == ['feature1', 'feature2']
    assert 'label' not in df.columns
    assert len(df) >= 40


def test_anomaly_fixture_contract():
    df = _load('anomaly.csv')
    assert list(df.columns) == ['feature1', 'feature2']
    assert 'label' not in df.columns
    assert len(df) >= 40
    # A small injected outlier group sits well beyond the inlier spread.
    radii = (df**2).sum(axis=1) ** 0.5
    core = radii.quantile(0.9)
    assert (radii > 3 * core).sum() < len(df) / 5


def test_invalid_fixtures():
    assert (FIXTURES / 'empty.csv').stat().st_size == 0

    header_only = _load('header_only.csv')
    assert list(header_only.columns) == ['feature1', 'feature2', 'label']
    assert header_only.empty

    non_numeric = _load('non_numeric_y.csv')
    assert not pd.api.types.is_numeric_dtype(non_numeric['label'])


def test_semi_supervised_fixture_trains():
    df = _load('semi_supervised.csv')
    _, metrics, _ = train_pipeline(_job('self_training_classifier', 'semi_supervised', 'label'), df)
    assert {'accuracy', 'precisionMacro', 'recallMacro', 'f1Macro'} <= set(metrics)
    assert metrics['labelledRows'] >= 40
    assert metrics['unlabelledRows'] >= 10
    assert metrics['iterations'] >= 1


def test_clustering_fixture_trains_with_separated_blobs():
    df = _load('clustering.csv')
    _, metrics, _ = train_pipeline(_job('kmeans', 'clustering', None, {'n_clusters': 3}), df)
    assert {'silhouette', 'daviesBouldin', 'calinskiHarabasz', 'inertia'} <= set(metrics)
    # Three well-separated blobs should cluster cleanly.
    assert metrics['silhouette'] > 0.6


def test_anomaly_fixture_flags_the_injected_outliers():
    df = _load('anomaly.csv')
    _, metrics, _ = train_pipeline(
        _job('isolation_forest', 'anomaly_detection', None, {'contamination': 0.1}), df
    )
    assert {'anomalyCount', 'anomalyRate', 'scoreMean', 'scoreStd'} <= set(metrics)
    assert metrics['anomalyCount'] == 6


@pytest.mark.parametrize(
    ('fixture', 'algorithm_id', 'task_type'),
    [
        ('binary_class.csv', 'logistic_regression', 'classification'),
        ('regression.csv', 'linear_regression', 'regression'),
    ],
)
def test_supervised_fixtures_train(fixture: str, algorithm_id: str, task_type: str):
    df = _load(fixture)
    _, metrics, _ = train_pipeline(_job(algorithm_id, task_type, 'label'), df)
    if task_type == 'classification':
        assert {'accuracy', 'precisionMacro', 'recallMacro', 'f1Macro'} <= set(metrics)
    else:
        assert {'rmse', 'mae', 'r2'} <= set(metrics)
