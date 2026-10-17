"""Catalogue integrity: ids, task split, and entry shape (FR-020, FR-021).

No server, no database. The id sets mirror the catalogue table in
``docs/01-requirements/functional-requirements.md`` — keep them in sync.
Pipeline/export parity lives in ``tests/training/core/test_pipeline_export.py``.
"""

from __future__ import annotations

import json

from src.training.catalogue import ALGORITHMS
from tests.training.fixtures import (
    ANOMALY_IDS,
    CLASSIFICATION_IDS,
    CLUSTERING_IDS,
    REGRESSION_IDS,
    RL_IDS,
    SEMI_SUPERVISED_IDS,
)


def test_catalogue_matches_spec():
    """Catalogue ids and task split mirror the spec table (FR-020, FR-021)."""
    expected = (
        CLASSIFICATION_IDS
        | REGRESSION_IDS
        | SEMI_SUPERVISED_IDS
        | CLUSTERING_IDS
        | ANOMALY_IDS
        | RL_IDS
    )
    assert set(ALGORITHMS) == expected
    for algo_id in CLASSIFICATION_IDS:
        assert ALGORITHMS[algo_id]['task_type'] == 'classification'
    for algo_id in REGRESSION_IDS:
        assert ALGORITHMS[algo_id]['task_type'] == 'regression'
    for algo_id in SEMI_SUPERVISED_IDS:
        assert ALGORITHMS[algo_id]['task_type'] == 'semi_supervised'
    for algo_id in CLUSTERING_IDS:
        assert ALGORITHMS[algo_id]['task_type'] == 'clustering'
    for algo_id in ANOMALY_IDS:
        assert ALGORITHMS[algo_id]['task_type'] == 'anomaly_detection'
    for algo_id in RL_IDS:
        assert ALGORITHMS[algo_id]['task_type'] == 'reinforcement_learning'


def test_catalogue_entries_are_complete():
    """Every entry exposes the fields documented for the catalogue (FR-020)."""
    for spec in ALGORITHMS.values():
        assert spec['name']
        assert spec['description']
        assert spec['family'] == 'sklearn'
        assert spec['task_type'] in {
            'classification',
            'regression',
            'semi_supervised',
            'clustering',
            'anomaly_detection',
            'reinforcement_learning',
        }
        assert isinstance(spec['exports'], list) and all(
            e in {'onnx', 'pkl', 'joblib'} for e in spec['exports']
        )
        # The API serialises defaults, so they must be JSON-encodable.
        json.dumps(spec['default_hyperparameters'])
        assert spec['estimator'] is not None
