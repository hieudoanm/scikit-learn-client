"""Unit tests for job task/label/RL column validation (FR-011/090/111)."""

import uuid
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from src.api.jobs_lifecycle import create_job
from src.api.schemas import CreateJobRequest, JobAcceptedResponse, TaskType
from tests.api.jobs_helpers import single_dataset_result


@pytest.mark.asyncio
async def test_create_job_invalid_task_type(mock_session, sample_dataset):
    """Test job creation with invalid task type."""
    single_dataset_result(mock_session, sample_dataset)

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1'],
        labelColumn='label',
        taskType=TaskType.CLASSIFICATION,  # Use valid task type but invalid algorithm
        algorithmId='unknown_algo',
    )

    with pytest.raises(Exception) as exc_info:
        await create_job(mock_session, body)
    # Unknown algorithm is checked before task type
    assert 'UNKNOWN_ALGORITHM' in str(exc_info.value) or (
        'TASK_ALGORITHM_MISMATCH' in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_create_job_invalid_column_mapping(mock_session, sample_dataset):
    """Test job creation with invalid column mapping."""
    single_dataset_result(mock_session, sample_dataset)

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['nonexistent_column'],
        labelColumn='label',
        taskType=TaskType.CLASSIFICATION,
        algorithmId='logistic_regression',
    )

    with pytest.raises(Exception) as exc_info:
        await create_job(mock_session, body)
    assert 'INVALID_COLUMN_MAP' in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_job_regression_label_type_check_bypass(
    mock_session, sample_dataset, monkeypatch
):
    """Test that regression label type check doesn't block on file read errors."""
    single_dataset_result(mock_session, sample_dataset)

    # Mock dataset_path to raise an exception (simulating file read error)
    mock_dataset_path = MagicMock(side_effect=FileNotFoundError('File not found'))
    monkeypatch.setattr('src.api.jobs_validation.dataset_path', mock_dataset_path)

    # Mock enqueue_job
    mock_enqueue = MagicMock()
    monkeypatch.setattr('src.api.jobs_lifecycle.enqueue_job', mock_enqueue)

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1'],
        labelColumn='label',
        taskType=TaskType.REGRESSION,
        algorithmId='linear_regression',
    )

    # Should succeed despite file read error (exception is caught and ignored)
    result = await create_job(mock_session, body)
    assert isinstance(result, JobAcceptedResponse)
    assert result.status.value == 'queued'
    mock_enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_create_job_clustering_without_label(mock_session, sample_dataset, monkeypatch):
    """Unsupervised jobs omit labelColumn (FR-090)."""
    sample_dataset.row_count = 100
    single_dataset_result(mock_session, sample_dataset)
    monkeypatch.setattr('src.api.jobs_lifecycle.enqueue_job', MagicMock())

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1', 'feature2'],
        taskType=TaskType.CLUSTERING,
        algorithmId='kmeans',
    )

    result = await create_job(mock_session, body)

    assert isinstance(result, JobAcceptedResponse)
    assert result.taskType == TaskType.CLUSTERING
    assert result.algorithmId == 'kmeans'


@pytest.mark.asyncio
async def test_create_job_supervised_without_label_rejected():
    """Supervised and semi-supervised jobs require labelColumn (FR-011) - validated by Pydantic."""
    with pytest.raises(ValidationError) as exc_info:
        CreateJobRequest(
            datasetId=uuid.uuid4(),
            featureColumns=['feature1'],
            taskType=TaskType.CLASSIFICATION,
            algorithmId='logistic_regression',
        )
    assert 'labelColumn required for classification' in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_job_unsupervised_with_label_rejected():
    """Clustering and anomaly jobs must not set labelColumn (FR-090) - validated by Pydantic."""
    with pytest.raises(ValidationError) as exc_info:
        CreateJobRequest(
            datasetId=uuid.uuid4(),
            featureColumns=['feature1', 'feature2'],
            labelColumn='label',
            taskType=TaskType.CLUSTERING,
            algorithmId='kmeans',
        )
    assert 'must not set labelColumn' in str(exc_info.value) or (
        'clustering must not set' in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_create_job_invalid_hyperparameters(mock_session, sample_dataset):
    """Out-of-range hyperparameters return INVALID_HYPERPARAMETERS (FR-027)."""
    sample_dataset.row_count = 100
    single_dataset_result(mock_session, sample_dataset)

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1', 'feature2'],
        taskType=TaskType.CLUSTERING,
        algorithmId='kmeans',
        hyperparameters={'n_clusters': 1},
    )

    with pytest.raises(Exception) as exc_info:
        await create_job(mock_session, body)
    assert 'INVALID_HYPERPARAMETERS' in str(exc_info.value)
