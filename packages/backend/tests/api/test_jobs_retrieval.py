"""Unit tests for job listing, polling, deletion, and artefact download."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.api.jobs import list_jobs
from src.api.jobs_lifecycle import delete_job, download_artefact, get_job
from src.api.schemas import JobAcceptedResponse, JobListResponse, JobResponse, JobStatus
from starlette.requests import Request


def _request(query: str = '') -> Request:
    """Build a minimal Starlette request with the given query string."""
    return Request(
        {
            'type': 'http',
            'method': 'GET',
            'path': '/api/v1/jobs',
            'raw_path': b'/api/v1/jobs',
            'query_string': query.encode(),
            'headers': [],
            'client': ('testclient', 123),
            'server': ('testserver', 80),
        }
    )


@pytest.mark.asyncio
async def test_list_jobs_success(mock_session, sample_job):
    """Test successful job listing."""
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[sample_job]))
    )

    # `list_jobs` issues one page query plus one queue-position query.
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await list_jobs(mock_session, _request(), limit=20)

    assert isinstance(result, JobListResponse)
    assert result.nextCursor is None
    assert result.hasMore is False
    assert len(result.jobs) == 1
    assert isinstance(result.jobs[0], JobAcceptedResponse)
    assert result.jobs[0].tags == ['baseline']
    assert result.jobs[0].attempts == 1
    assert result.jobs[0].maxAttempts == 2


@pytest.mark.asyncio
async def test_list_jobs_invalid_limit(mock_session):
    """Test job listing with invalid limit."""
    with pytest.raises(Exception) as exc_info:
        await list_jobs(mock_session, _request(), limit=0)
    assert 'VALIDATION_ERROR' in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        await list_jobs(mock_session, _request(), limit=101)
    assert 'VALIDATION_ERROR' in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_jobs_invalid_cursor(mock_session):
    """Test job listing with invalid cursor."""
    with pytest.raises(Exception) as exc_info:
        await list_jobs(mock_session, _request(), cursor='invalid-uuid')
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'Invalid cursor' in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_jobs_rejects_repeated_query_param(mock_session):
    """A repeated query parameter is rejected (the comma-separated form wins)."""
    with pytest.raises(Exception) as exc_info:
        await list_jobs(mock_session, _request('status=queued&status=failed'))
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'Repeated query parameter' in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_job_success(mock_session, sample_job):
    """Test successful job retrieval."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=sample_job)
    mock_session.execute = AsyncMock(return_value=mock_result)

    # Mock artefacts and metrics queries
    mock_artefacts_result = MagicMock()
    mock_artefacts_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[]))
    )
    mock_metrics_result = MagicMock()
    mock_metrics_result.scalars = MagicMock(
        return_value=MagicMock(first=MagicMock(return_value=None))
    )

    mock_session.execute = AsyncMock(
        side_effect=[mock_result, mock_artefacts_result, mock_metrics_result]
    )

    result = await get_job(mock_session, sample_job.id)

    assert isinstance(result, JobResponse)
    assert str(result.jobId) == str(sample_job.id)
    assert result.status == JobStatus.COMPLETED
    assert result.artefacts == []
    assert result.metrics is None


@pytest.mark.asyncio
async def test_get_job_not_found(mock_session):
    """Test job retrieval when job doesn't exist."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')

    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(Exception) as exc_info:
        await get_job(mock_session, job_id)
    assert 'JOB_NOT_FOUND' in str(exc_info.value)


@pytest.mark.asyncio
async def test_delete_job_success(mock_session, sample_job, monkeypatch):
    """Test successful job deletion."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=sample_job)
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_remove_files = MagicMock()
    monkeypatch.setattr('src.api.jobs_lifecycle.remove_job_files', mock_remove_files)
    mock_cancel = MagicMock()
    monkeypatch.setattr('src.api.jobs_lifecycle.cancel_job', mock_cancel)

    result = await delete_job(mock_session, sample_job.id)

    assert result.status_code == 204
    mock_remove_files.assert_called_once_with(sample_job.id)
    mock_cancel.assert_called_once_with(sample_job.id)
    # execute is called 4 times: select Job, delete JobMetric, delete Artefact, delete Job
    assert mock_session.execute.call_count == 4


@pytest.mark.asyncio
async def test_delete_job_not_found(mock_session):
    """Test job deletion when job doesn't exist."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')

    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(Exception) as exc_info:
        await delete_job(mock_session, job_id)
    assert 'JOB_NOT_FOUND' in str(exc_info.value)


@pytest.mark.asyncio
async def test_download_artefact_success(mock_session, sample_job, monkeypatch, tmp_path):
    """Test successful artefact download."""
    artefact_id = uuid.UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')

    # Mock artefact
    artefact = MagicMock()
    artefact.id = artefact_id
    artefact.format = 'onnx'
    artefact.filename = 'model.onnx'
    artefact.storage_key = 'artefacts/123/model.onnx'

    # Create a temporary file
    test_file = tmp_path / 'model.onnx'
    test_file.write_bytes(b'model data')

    mock_artefact_result = MagicMock()
    mock_artefact_result.scalar_one_or_none = MagicMock(return_value=artefact)

    mock_job_result = MagicMock()
    mock_job_result.scalar_one_or_none = MagicMock(return_value=sample_job)

    mock_session.execute = AsyncMock(side_effect=[mock_artefact_result, mock_job_result])

    mock_artefact_path = MagicMock(return_value=test_file)
    monkeypatch.setattr('src.api.jobs_lifecycle.artefact_path', mock_artefact_path)

    result = await download_artefact(mock_session, sample_job.id, artefact_id)

    assert result.filename == 'model.onnx'
    assert result.media_type == 'application/octet-stream'


@pytest.mark.asyncio
async def test_download_artefact_not_found(mock_session):
    """Test artefact download when artefact doesn't exist."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    artefact_id = uuid.UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')

    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(Exception) as exc_info:
        await download_artefact(mock_session, job_id, artefact_id)
    assert 'ARTEFACT_NOT_FOUND' in str(exc_info.value)


@pytest.mark.asyncio
async def test_download_artefact_job_not_completed(mock_session, sample_job):
    """Test artefact download when job is not completed."""
    artefact_id = uuid.UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')

    artefact = MagicMock()
    artefact.id = artefact_id
    artefact.format = 'onnx'
    artefact.filename = 'model.onnx'
    artefact.storage_key = 'artefacts/123/model.onnx'

    sample_job.status = 'training'  # Not completed

    mock_artefact_result = MagicMock()
    mock_artefact_result.scalar_one_or_none = MagicMock(return_value=artefact)

    mock_job_result = MagicMock()
    mock_job_result.scalar_one_or_none = MagicMock(return_value=sample_job)

    mock_session.execute = AsyncMock(side_effect=[mock_artefact_result, mock_job_result])

    with pytest.raises(Exception) as exc_info:
        await download_artefact(mock_session, sample_job.id, artefact_id)
    assert 'ARTEFACT_NOT_FOUND' in str(exc_info.value)
