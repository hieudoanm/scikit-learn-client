"""Unit tests for job serialization (src/api/jobs_serialization.py)."""

import uuid
from unittest.mock import MagicMock

from src.api.jobs import JOB_STATUS_ORDER
from src.api.jobs_serialization import _job_accepted_response, _job_detail_response


def test_job_status_order():
    """Test that JOB_STATUS_ORDER is defined correctly."""
    assert JOB_STATUS_ORDER == [
        'queued',
        'validating',
        'training',
        'evaluating',
        'exporting',
        'retrying',
        'completed',
        'failed',
    ]


def test_job_accepted_response_basic(sample_job):
    """Test _job_accepted_response with basic job."""
    result = _job_accepted_response(sample_job)
    assert str(result.jobId) == str(sample_job.id)
    assert result.status.value == 'completed'
    assert str(result.datasetId) == str(sample_job.dataset_id)
    assert result.taskType.value == 'classification'
    assert result.algorithmId == 'logistic_regression'
    assert isinstance(result.createdAt, object)  # datetime


def test_job_detail_response_basic(sample_job):
    """Test _job_detail_response with basic job."""
    # Mock artefacts and metrics
    artefact = MagicMock()
    artefact.id = uuid.UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
    artefact.format = 'onnx'
    artefact.filename = 'model.onnx'
    artefact.byte_size = 1024

    metric = MagicMock()
    metric.metrics = {'accuracy': 0.95, 'precisionMacro': 0.93}
    metric.charts = {'classes': ['0', '1']}

    sample_job.artefacts = [artefact]

    result = _job_detail_response(sample_job, [artefact], metric)
    assert str(result.jobId) == str(sample_job.id)
    assert result.status.value == 'completed'
    assert str(result.datasetId) == str(sample_job.dataset_id)
    assert result.taskType.value == 'classification'
    assert result.algorithmId == 'logistic_regression'
    assert result.featureColumns == ['feature1', 'feature2']
    assert result.labelColumn == 'label'
    assert result.hyperparameters == {'C': 1.0}
    assert result.libraryVersions == {'scikit-learn': '1.9.1'}
    assert result.error is None
    assert result.metrics == {'accuracy': 0.95, 'precisionMacro': 0.93}
    assert result.charts == {'classes': ['0', '1']}
    assert len(result.artefacts) == 1
    assert str(result.artefacts[0].artefactId) == str(artefact.id)
    assert result.artefacts[0].format.value == 'onnx'


def test_job_detail_response_with_error(sample_job):
    """Test _job_detail_response with error."""
    sample_job.error_code = 'TRAINING_FAILED'
    sample_job.error_message = 'Training error occurred'

    result = _job_detail_response(sample_job, [], None)
    assert result.error is not None
    assert result.error.code == 'TRAINING_FAILED'
    assert result.error.message == 'Training error occurred'


def test_job_detail_response_rl_columns(sample_job):
    """RL jobs expose rlColumns and a null labelColumn (FEATURE-11)."""
    sample_job.task_type = 'reinforcement_learning'
    sample_job.label_column = None
    sample_job.rl_columns = {
        'actionColumn': 'action',
        'rewardColumn': 'reward',
        'episodeColumn': 'episode',
    }

    result = _job_detail_response(sample_job, [], None)
    assert result.labelColumn is None
    assert result.rlColumns is not None
    assert result.rlColumns.actionColumn == 'action'
    assert result.rlColumns.rewardColumn == 'reward'
    assert result.rlColumns.episodeColumn == 'episode'
