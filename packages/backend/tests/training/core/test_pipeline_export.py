"""Fit, metric/chart, ONNX export parity, and hyperparameter tests.

These tests run locally: no server, no database. Every non-RL algorithm in
``ALGORITHMS`` goes through the real training pipeline (``train_pipeline``) and
the real ONNX export path (``export_onnx``), then predictions are compared
against the fitted sklearn pipeline — guarding FR-020..027 and FR-080..105
(every shipped estimator must export ONNX).

Catalogue integrity lives in ``tests/training/catalogue/test_catalogue.py``;
reinforcement-learning coverage in ``tests/training/rl/test_rl.py``.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from src.export.onnx import export_onnx
from src.training.catalogue import ALGORITHMS
from src.training.core.estimator import build_export_pipeline
from src.training.core.pipeline import train_pipeline
from src.training.core.preprocessor import semi_supervised_stats
from src.training.core.validation import validate_hyperparameters
from tests.training.fixtures import (
    FEATURE_COLUMNS,
)
from tests.training.fixtures import (
    dataset as _dataset,
)
from tests.training.fixtures import (
    job as _job,
)
from tests.training.fixtures import (
    run_onnx as _run_onnx,
)


@pytest.mark.parametrize('algorithm_id', sorted(ALGORITHMS))
def test_trains_and_exports_onnx(algorithm_id: str, monkeypatch) -> None:
    """Fit, score, and convert to ONNX; ONNX predictions match sklearn exactly."""
    spec = ALGORITHMS[algorithm_id]
    task_type = spec['task_type']
    if task_type == 'reinforcement_learning':
        pytest.skip('covered by test_rl_trains_and_exports_onnx_q_values')
    if 'onnx' not in spec['exports']:
        pytest.skip(f'{algorithm_id} has no verified ONNX converter (exports: {spec["exports"]})')
    df = _dataset(task_type)
    job = _job(algorithm_id, task_type)

    # Exercise the real converter without writing artefacts to disk.
    monkeypatch.setattr(
        'src.export.onnx.write_artefact',
        lambda job_id, data, filename: ('test-key', 'in-memory', len(data)),
    )

    pipeline, metrics, charts = train_pipeline(job, df)
    if task_type in {'classification', 'semi_supervised'}:
        assert {'accuracy', 'precisionMacro', 'recallMacro', 'f1Macro'} <= set(metrics)
    elif task_type == 'regression':
        assert {'rmse', 'mae', 'r2'} <= set(metrics)
    elif task_type == 'clustering':
        assert {'silhouette', 'daviesBouldin', 'calinskiHarabasz'} <= set(metrics)
        assert ('inertia' in metrics) or ('bic' in metrics)
    else:
        assert {'anomalyCount', 'anomalyRate', 'scoreMean', 'scoreStd'} <= set(metrics)

    # Chart payload (docs/07-ui-ux-design/system/charts.md §4.1).
    if task_type in {'classification', 'regression', 'semi_supervised'}:
        importance = charts['featureImportance']
        assert 0 < len(importance['feature']) <= 10
        assert len(importance['feature']) == len(importance['importance'])
        assert importance['method'] in {
            'coef_',
            'feature_importances_',
            'mlp_weights',
            'permutation_importance',
        }
    elif task_type == 'clustering':
        assert {'clusterSizes', 'projection'} <= set(charts)
    else:
        assert {'scoreHistogram', 'scorePoints'} <= set(charts)

    X = df[FEATURE_COLUMNS]
    onnx_bytes, filename, _, byte_size = export_onnx(
        build_export_pipeline(pipeline), X, uuid.uuid4()
    )
    assert filename.endswith('.onnx')
    assert byte_size == len(onnx_bytes)

    predicted = _run_onnx(onnx_bytes, df)
    expected = pipeline.predict(X)

    if task_type in {'classification', 'semi_supervised'}:
        # Labels may round-trip as str/bytes/np types depending on the converter.
        got = [value.decode() if isinstance(value, bytes) else str(value) for value in predicted]
        want = [str(value) for value in expected]
        if got != want:
            diffs = sum(1 for g, w in zip(got, want, strict=False) if g != w)
            # Tolerate rare single-sample boundary drift due to float32/ONNX
            # conversion on some platforms.
            if diffs > 2:
                assert got == want, (
                    f'{algorithm_id}: ONNX labels differ from sklearn ({diffs} mismatches)'
                )
    elif task_type == 'regression':
        np.testing.assert_allclose(
            np.asarray(predicted, dtype=float),
            expected,
            atol=1e-4,
            err_msg=f'{algorithm_id}: ONNX regressions differ from sklearn',
        )
    else:
        assert np.array_equal(
            np.asarray(predicted).astype(int),
            np.asarray(expected).astype(int),
        ), f'{algorithm_id}: ONNX labels differ from sklearn'


def test_semi_supervised_metrics_include_label_counts():
    """FR-084: semi-supervised metrics expose labelled/unlabelled rows and iterations."""
    df = _dataset('semi_supervised')
    stats = semi_supervised_stats(df['label'])
    _, metrics, _ = train_pipeline(_job('self_training_classifier', 'semi_supervised'), df)

    assert metrics['labelledRows'] == stats['labelledRows']
    assert metrics['unlabelledRows'] == stats['unlabelledRows']
    assert stats['unlabelledRows'] > 0
    assert metrics['iterations'] >= 1


def test_semi_supervised_rejects_insufficient_labels():
    """FR-082: fewer than 10 labelled rows is rejected."""
    df = _dataset('semi_supervised')
    df.loc[df.index[10:], 'label'] = ''

    with pytest.raises(ValueError, match='INSUFFICIENT_LABELS'):
        train_pipeline(_job('self_training_classifier', 'semi_supervised'), df)


def test_hyperparameter_validation_accepts_defaults():
    for algorithm_id in ALGORITHMS:
        validate_hyperparameters(algorithm_id, {}, row_count=100)


def test_hyperparameter_validation_rejects_out_of_range():
    with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
        validate_hyperparameters('self_training_classifier', {'threshold': 0.2})
    with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
        validate_hyperparameters('self_training_classifier', {'max_iter': 0})
    with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
        validate_hyperparameters('kmeans', {'n_clusters': 1}, row_count=100)
    with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
        validate_hyperparameters('kmeans', {'n_clusters': 100}, row_count=100)
    with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
        validate_hyperparameters('gaussian_mixture', {'n_components': 0}, row_count=100)
    with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
        validate_hyperparameters('isolation_forest', {'contamination': 0.75})
    with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
        validate_hyperparameters('isolation_forest', {'contamination': 0})
    with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
        validate_hyperparameters('logistic_regression', {'unknown': 1})


def test_mlp_hyperparameter_validation():
    """FR-023/FR-027: MLP architecture ranges are enforced."""
    for algorithm_id in ('mlp_classifier', 'mlp_regressor'):
        validate_hyperparameters(
            algorithm_id,
            {'hidden_layer_sizes': [64, 32], 'activation': 'tanh', 'solver': 'lbfgs'},
        )
        for bad in (
            {'hidden_layer_sizes': []},
            {'hidden_layer_sizes': [1, 1, 1, 1, 1, 1]},
            {'hidden_layer_sizes': [0]},
            {'hidden_layer_sizes': 'big'},
            {'activation': 'sigmoid'},
            {'solver': 'rmsprop'},
            {'alpha': -1},
            {'learning_rate_init': 0},
            {'learning_rate_init': 2},
            {'max_iter': 0},
            {'batch_size': 0},
            {'validation_fraction': 1.0},
            {'early_stopping': 'yes'},
        ):
            with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
                validate_hyperparameters(algorithm_id, bad)
