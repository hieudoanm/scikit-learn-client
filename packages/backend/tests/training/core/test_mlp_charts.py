"""MLP chart coverage (docs/07-ui-ux-design/system/charts.md §3.6, §3.8).

Neural-net jobs keep their task family's charts and add ``mlpProgress`` when the
solver exposes a loss curve; feature importance for an MLP is derived from the
first hidden layer (``coefs_[0]``) rather than the permutation fallback.
"""

from __future__ import annotations

import pytest
from src.training.core.pipeline import train_pipeline
from src.training.core.validation import validate_hyperparameters
from tests.training.fixtures import (
    dataset as _dataset,
)
from tests.training.fixtures import (
    job as _job,
)


@pytest.mark.parametrize(
    ('algorithm_id', 'task_type'),
    [('mlp_classifier', 'classification'), ('mlp_regressor', 'regression')],
)
def test_mlp_feature_importance_uses_first_layer(algorithm_id: str, task_type: str) -> None:
    """M1 for an MLP comes from coefs_[0], not permutation importance."""
    _, _, charts = train_pipeline(_job(algorithm_id, task_type), _dataset(task_type))

    importance = charts['featureImportance']
    assert importance['method'] == 'mlp_weights'
    assert 0 < len(importance['feature']) <= 10
    assert len(importance['feature']) == len(importance['importance'])


@pytest.mark.parametrize(
    ('algorithm_id', 'task_type'),
    [('mlp_classifier', 'classification'), ('mlp_regressor', 'regression')],
)
def test_mlp_progress_present_for_adam(algorithm_id: str, task_type: str) -> None:
    """MLP1: adam/sgd expose a per-epoch loss curve."""
    job = _job(algorithm_id, task_type, hyperparameters={'solver': 'adam', 'max_iter': 5})
    _, _, charts = train_pipeline(job, _dataset(task_type))

    progress = charts['mlpProgress']
    assert progress['epoch'] == list(range(1, len(progress['loss']) + 1))
    assert 0 < len(progress['loss']) <= 5
    assert all(isinstance(value, float) for value in progress['loss'])


@pytest.mark.parametrize(
    ('algorithm_id', 'task_type'),
    [('mlp_classifier', 'classification'), ('mlp_regressor', 'regression')],
)
def test_mlp_progress_absent_for_lbfgs(algorithm_id: str, task_type: str) -> None:
    """MLP1 is hidden when the solver has no loss curve (lbfgs)."""
    job = _job(algorithm_id, task_type, hyperparameters={'solver': 'lbfgs', 'max_iter': 20})
    _, _, charts = train_pipeline(job, _dataset(task_type))

    assert 'mlpProgress' not in charts


def test_non_mlp_jobs_have_no_mlp_progress() -> None:
    """MLP1 is an MLP-only addition; other estimators are unaffected."""
    _, _, charts = train_pipeline(
        _job('random_forest_classifier', 'classification'), _dataset('classification')
    )
    assert 'mlpProgress' not in charts


def test_mlp_defaults_are_valid() -> None:
    """The enriched catalogue defaults pass validation unchanged."""
    for algorithm_id in ('mlp_classifier', 'mlp_regressor'):
        validate_hyperparameters(algorithm_id, {}, row_count=120)
