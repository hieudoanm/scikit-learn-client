"""Unit tests for src/training/worker/worker.py helper functions."""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import src.training.worker.worker as worker
from src.db.models import Job
from src.training.worker import _validate_mapping, cancel_job, is_terminal
from src.training.worker.store import JobFailure


def test_is_terminal_with_completed():
    """Test is_terminal with completed status."""
    assert is_terminal('completed') is True


def test_is_terminal_with_failed():
    """Test is_terminal with failed status."""
    assert is_terminal('failed') is True


def test_is_terminal_with_non_terminal_status():
    """Test is_terminal with non-terminal statuses."""
    assert is_terminal('queued') is False
    assert is_terminal('validating') is False
    assert is_terminal('training') is False
    assert is_terminal('evaluating') is False
    assert is_terminal('exporting') is False


def test_validate_mapping_valid():
    """Test _validate_mapping with valid mapping."""
    df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'label': [0, 1]})
    job = Job(
        feature_columns=['feature1', 'feature2'],
        label_column='label',
        task_type='classification',
        algorithm_id='logistic_regression',
    )
    _validate_mapping(df, job)  # Should not raise


def test_validate_mapping_missing_feature_columns():
    """Test _validate_mapping with missing feature columns."""
    df = pd.DataFrame({'feature1': [1, 2], 'label': [0, 1]})
    job = Job(
        feature_columns=['feature1', 'feature2'],
        label_column='label',
        task_type='classification',
        algorithm_id='logistic_regression',
    )
    with pytest.raises(ValueError) as exc_info:
        _validate_mapping(df, job)
    assert 'INVALID_COLUMN_MAP' in str(exc_info.value)
    assert 'feature2' in str(exc_info.value)


def test_validate_mapping_missing_label_column():
    """Test _validate_mapping with missing label column."""
    df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    job = Job(
        feature_columns=['feature1', 'feature2'],
        label_column='label',
        task_type='classification',
        algorithm_id='logistic_regression',
    )
    with pytest.raises(ValueError) as exc_info:
        _validate_mapping(df, job)
    assert 'INVALID_COLUMN_MAP' in str(exc_info.value)
    assert 'label' in str(exc_info.value)


def test_validate_mapping_label_in_features():
    """Test _validate_mapping when label column is also a feature column."""
    df = pd.DataFrame({'feature1': [1, 2], 'label': [0, 1]})
    job = Job(
        feature_columns=['feature1', 'label'],
        label_column='label',
        task_type='classification',
        algorithm_id='logistic_regression',
    )
    with pytest.raises(ValueError) as exc_info:
        _validate_mapping(df, job)
    assert 'INVALID_COLUMN_MAP' in str(exc_info.value)
    assert 'label' in str(exc_info.value)


def test_validate_mapping_regression_numeric_label():
    """Test _validate_mapping with regression and numeric label."""
    df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'label': [0.5, 1.5]})
    job = Job(
        feature_columns=['feature1', 'feature2'],
        label_column='label',
        task_type='regression',
        algorithm_id='linear_regression',
    )
    _validate_mapping(df, job)  # Should not raise


def test_validate_mapping_regression_non_numeric_label():
    """Test _validate_mapping with regression and non-numeric label."""
    df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'label': ['a', 'b']})
    job = Job(
        feature_columns=['feature1', 'feature2'],
        label_column='label',
        task_type='regression',
        algorithm_id='linear_regression',
    )
    with pytest.raises(ValueError) as exc_info:
        _validate_mapping(df, job)
    assert 'INVALID_LABEL_TYPE' in str(exc_info.value)
    assert 'label' in str(exc_info.value)
    assert 'numeric' in str(exc_info.value)


def test_validate_mapping_classification_any_label():
    """Test _validate_mapping with classification (label type not checked)."""
    df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'label': ['a', 'b']})
    job = Job(
        feature_columns=['feature1', 'feature2'],
        label_column='label',
        task_type='classification',
        algorithm_id='logistic_regression',
    )
    _validate_mapping(df, job)  # Should not raise even with string label


def test_validate_mapping_clustering_without_label():
    """Unsupervised jobs validate without a label column (FR-090)."""
    df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    job = Job(
        feature_columns=['feature1', 'feature2'],
        label_column=None,
        task_type='clustering',
        algorithm_id='kmeans',
    )
    _validate_mapping(df, job)  # Should not raise


def test_validate_mapping_clustering_with_label_rejected():
    """Unsupervised jobs reject a label column (FR-090)."""
    df = pd.DataFrame({'feature1': [1, 2], 'label': [0, 1]})
    job = Job(
        feature_columns=['feature1'],
        label_column='label',
        task_type='anomaly_detection',
        algorithm_id='isolation_forest',
    )
    with pytest.raises(ValueError) as exc_info:
        _validate_mapping(df, job)
    assert 'INVALID_COLUMN_MAP' in str(exc_info.value)


def test_validate_mapping_supervised_without_label_rejected():
    """Supervised jobs require a label column (FR-011)."""
    df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    job = Job(
        feature_columns=['feature1', 'feature2'],
        label_column=None,
        task_type='classification',
        algorithm_id='logistic_regression',
    )
    with pytest.raises(ValueError) as exc_info:
        _validate_mapping(df, job)
    assert 'INVALID_COLUMN_MAP' in str(exc_info.value)


def test_validate_mapping_semi_supervised_valid():
    """Semi-supervised jobs allow empty and whitespace-only label cells (FR-081)."""
    labels = ['yes', 'no', ''] * 4 + ['yes', 'no', '  ', 'yes']
    df = pd.DataFrame({'feature1': range(len(labels)), 'label': labels})
    job = Job(
        feature_columns=['feature1'],
        label_column='label',
        task_type='semi_supervised',
        algorithm_id='self_training_classifier',
    )
    _validate_mapping(df, job)  # Should not raise


def test_validate_mapping_semi_supervised_insufficient_labels():
    """Fewer than 10 labelled rows is rejected at training time (FR-082)."""
    df = pd.DataFrame({'feature1': range(20), 'label': ['yes'] * 4 + [''] * 16})
    job = Job(
        feature_columns=['feature1'],
        label_column='label',
        task_type='semi_supervised',
        algorithm_id='self_training_classifier',
    )
    with pytest.raises(ValueError) as exc_info:
        _validate_mapping(df, job)
    assert 'INSUFFICIENT_LABELS' in str(exc_info.value)


def test_cancel_job_is_noop_for_unknown_job():
    """Cancelling a job with no registered task does nothing."""
    cancel_job(uuid.uuid4())  # Should not raise


def test_job_failure_carries_code():
    """JobFailure exposes the FEATURE-07 code the worker records."""
    exc = JobFailure('EXPORT_FAILED', 'onnx conversion failed')
    assert exc.code == 'EXPORT_FAILED'
    assert str(exc) == 'onnx conversion failed'


def test_is_validation_error_recognises_documented_codes():
    """Validation codes surface as TRAINING_FAILED; anything else does not."""
    assert worker._is_validation_error('INVALID_COLUMN_MAP: X is also Y')
    assert worker._is_validation_error('INVALID_LABEL_TYPE: label not numeric')
    assert worker._is_validation_error('INSUFFICIENT_LABELS: need 10')
    assert not worker._is_validation_error('divide by zero')
    assert not worker._is_validation_error('')


def _fake_job(pipeline=None):
    job = Job(
        dataset_id=uuid.uuid4(),
        feature_columns=['feature1'],
        label_column='label',
        task_type='classification',
        algorithm_id='logistic_regression',
        export_pkl=False,
        export_joblib=False,
    )
    job.id = uuid.uuid4()
    return job


class _FakeSessionCM:
    """Async context manager yielding a fake AsyncSession-like object."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


def _patch_run_job(monkeypatch, job):
    """Patch the worker's collaborators so `_process_job` runs without a database."""
    session = AsyncMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=job))
    )

    recorded = {}

    async def fake_record_error(_session, _job_id, code, message):
        recorded['code'] = code
        recorded['message'] = message

    monkeypatch.setattr(worker, 'async_session', lambda: _FakeSessionCM(session))
    monkeypatch.setattr(
        worker,
        '_load_dataset_df',
        AsyncMock(return_value=pd.DataFrame({'feature1': [1.0, 2.0], 'label': [0, 1]})),
    )
    monkeypatch.setattr(worker, '_validate_mapping', MagicMock())
    monkeypatch.setattr(worker, '_set_status', AsyncMock())
    monkeypatch.setattr(worker, '_job_exists', AsyncMock(return_value=True))
    monkeypatch.setattr(worker, '_record_metrics', AsyncMock())
    monkeypatch.setattr(worker, '_record_artefact', AsyncMock())
    monkeypatch.setattr(worker, '_record_error', fake_record_error)
    return recorded


@pytest.mark.asyncio
async def test_run_job_records_export_failed(monkeypatch):
    """An export error is recorded as EXPORT_FAILED, not TRAINING_FAILED (FEATURE-07)."""
    job = _fake_job()
    recorded = _patch_run_job(monkeypatch, job)
    monkeypatch.setattr(worker, 'train_pipeline', MagicMock(return_value=(MagicMock(), {}, None)))
    monkeypatch.setattr(worker, 'build_export_pipeline', MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        worker, 'export_onnx', MagicMock(side_effect=RuntimeError('onnx conversion failed'))
    )

    await worker._process_job(job.id)

    assert recorded['code'] == 'EXPORT_FAILED'
    assert 'onnx conversion failed' in recorded['message']


@pytest.mark.asyncio
async def test_run_job_records_training_failed(monkeypatch):
    """A fit error is recorded as TRAINING_FAILED (FEATURE-07)."""
    job = _fake_job()
    recorded = _patch_run_job(monkeypatch, job)
    monkeypatch.setattr(
        worker, 'train_pipeline', MagicMock(side_effect=RuntimeError('fit blew up'))
    )

    await worker._process_job(job.id)

    assert recorded['code'] == 'TRAINING_FAILED'


@pytest.mark.asyncio
async def test_run_job_records_internal_error_for_unexpected(monkeypatch):
    """A non-validation, non-export error falls back to INTERNAL_ERROR."""
    job = _fake_job()
    recorded = _patch_run_job(monkeypatch, job)
    monkeypatch.setattr(worker, '_load_dataset_df', AsyncMock(side_effect=RuntimeError('boom')))

    await worker._process_job(job.id)

    assert recorded['code'] == 'INTERNAL_ERROR'


@pytest.mark.asyncio
async def test_run_job_records_training_failed_for_validation_codes(monkeypatch):
    """A mapping validation failure maps to TRAINING_FAILED."""
    job = _fake_job()
    recorded = _patch_run_job(monkeypatch, job)
    monkeypatch.setattr(
        worker,
        '_load_dataset_df',
        AsyncMock(side_effect=ValueError('INVALID_COLUMN_MAP: label also a feature')),
    )

    await worker._process_job(job.id)

    assert recorded['code'] == 'TRAINING_FAILED'


@pytest.mark.asyncio
async def test_cancel_job_cancels_registered_task():
    """Deleting a job cancels its in-process training task and deregisters it."""
    job_id = uuid.uuid4()
    task = asyncio.ensure_future(asyncio.sleep(60))
    worker._job_tasks[job_id] = task

    cancel_job(job_id)

    assert job_id not in worker._job_tasks
    with pytest.raises(asyncio.CancelledError):
        await task
