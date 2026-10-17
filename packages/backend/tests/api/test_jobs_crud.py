"""Unit tests for job creation (src/api/jobs.py)."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from src.api.jobs_lifecycle import create_job
from src.api.schemas import CreateJobRequest, JobAcceptedResponse, TaskType


@pytest.mark.asyncio
async def test_create_job_success(mock_session, sample_dataset, monkeypatch):
    """Test successful job creation."""
    # Mock database query
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=sample_dataset)
    mock_session.execute = AsyncMock(return_value=mock_result)

    # Mock enqueue_job
    mock_enqueue = MagicMock()
    monkeypatch.setattr('src.api.jobs_lifecycle.enqueue_job', mock_enqueue)

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1', 'feature2'],
        labelColumn='label',
        taskType=TaskType.CLASSIFICATION,
        algorithmId='logistic_regression',
    )

    result = await create_job(mock_session, body)

    assert isinstance(result, JobAcceptedResponse)
    assert result.status.value == 'queued'
    assert str(result.datasetId) == str(sample_dataset.id)
    assert result.taskType == TaskType.CLASSIFICATION
    assert result.algorithmId == 'logistic_regression'
    assert result.jobId is not None
    assert result.createdAt is not None
    mock_enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_create_job_missing_required_fields():
    """Test job creation with missing required fields - validated by Pydantic."""
    with pytest.raises(ValidationError) as exc_info:
        CreateJobRequest(
            datasetId=uuid.uuid4(),
            featureColumns=['feature1'],
            taskType=TaskType.CLASSIFICATION,
            algorithmId='logistic_regression',
        )
    assert 'labelColumn required for classification' in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_job_invalid_dataset_id():
    """Test job creation with invalid dataset ID - validated by Pydantic."""
    with pytest.raises(ValidationError) as exc_info:
        CreateJobRequest(
            datasetId='invalid-uuid',
            featureColumns=['feature1'],
            labelColumn='label',
            taskType=TaskType.CLASSIFICATION,
            algorithmId='logistic_regression',
        )
    assert 'uuid_parsing' in str(exc_info.value) or 'Invalid datasetId' in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_job_dataset_not_found(mock_session):
    """Test job creation when dataset doesn't exist."""
    dataset_id = uuid.UUID('12345678-1234-5678-1234-567812345678')

    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_session.execute = AsyncMock(return_value=mock_result)

    body = CreateJobRequest(
        datasetId=dataset_id,
        featureColumns=['feature1'],
        labelColumn='label',
        taskType=TaskType.CLASSIFICATION,
        algorithmId='logistic_regression',
    )

    with pytest.raises(Exception) as exc_info:
        await create_job(mock_session, body)
    assert 'DATASET_NOT_FOUND' in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_job_unknown_algorithm(mock_session, sample_dataset):
    """Test job creation with unknown algorithm."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=sample_dataset)
    mock_session.execute = AsyncMock(return_value=mock_result)

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1'],
        labelColumn='label',
        taskType=TaskType.CLASSIFICATION,
        algorithmId='unknown_algo',
    )

    with pytest.raises(Exception) as exc_info:
        await create_job(mock_session, body)
    assert 'UNKNOWN_ALGORITHM' in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_job_task_algorithm_mismatch(mock_session, sample_dataset):
    """Test job creation with task/algorithm mismatch."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=sample_dataset)
    mock_session.execute = AsyncMock(return_value=mock_result)

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1'],
        labelColumn='label',
        taskType=TaskType.REGRESSION,
        algorithmId='logistic_regression',  # Classification algorithm
    )

    with pytest.raises(Exception) as exc_info:
        await create_job(mock_session, body)
    assert 'TASK_ALGORITHM_MISMATCH' in str(exc_info.value)
