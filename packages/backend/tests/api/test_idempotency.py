"""Tests for `Idempotency-Key` handling on POST /api/v1/jobs (docs/06 §8)."""

import uuid
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from src.api import idempotency
from src.api.jobs_lifecycle import IDEMPOTENCY_KEY_PATTERN, create_job
from src.api.schemas import CreateJobRequest, TaskType
from src.db.utils import utc_now


def _record(**overrides):
    """Build a mock IdempotencyKey row."""
    record = MagicMock()
    record.key = overrides.get('key', 'k1')
    record.request_hash = overrides.get('request_hash', 'hash')
    record.job_id = overrides.get('job_id', uuid.uuid4())
    record.response_status = overrides.get('response_status', 202)
    record.response_body = overrides.get('response_body', {'jobId': 'cached'})
    record.created_at = overrides.get('created_at', utc_now())
    record.expires_at = overrides.get('expires_at', utc_now() + timedelta(hours=1))
    return record


def test_request_hash_is_order_independent():
    assert idempotency.request_hash({'a': 1, 'b': 2}) == idempotency.request_hash({'b': 2, 'a': 1})


def test_request_hash_differs_for_different_bodies():
    assert idempotency.request_hash({'a': 1}) != idempotency.request_hash({'a': 2})


def test_idempotency_key_pattern():
    assert IDEMPOTENCY_KEY_PATTERN.match('abc-123_XYZ')
    assert not IDEMPOTENCY_KEY_PATTERN.match('')
    assert not IDEMPOTENCY_KEY_PATTERN.match('has space')
    assert not IDEMPOTENCY_KEY_PATTERN.match('x' * 256)


@pytest.mark.asyncio
async def test_lookup_miss():
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result)

    assert await idempotency.lookup(session, 'k1', 'hash') is None


@pytest.mark.asyncio
async def test_lookup_replay():
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=_record(response_body={'jobId': 'j1'}))
    session.execute = AsyncMock(return_value=result)

    assert await idempotency.lookup(session, 'k1', 'hash') == {'jobId': 'j1'}


@pytest.mark.asyncio
async def test_lookup_expired_is_a_miss():
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(
        return_value=_record(expires_at=utc_now() - timedelta(seconds=1))
    )
    session.execute = AsyncMock(return_value=result)

    assert await idempotency.lookup(session, 'k1', 'hash') is None


@pytest.mark.asyncio
async def test_lookup_conflicting_payload_raises_409():
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=_record(request_hash='other'))
    session.execute = AsyncMock(return_value=result)

    with pytest.raises(HTTPException) as exc_info:
        await idempotency.lookup(session, 'k1', 'hash')

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail['error']['code'] == 'IDEMPOTENCY_KEY_EXISTS'


@pytest.mark.asyncio
async def test_create_job_rejects_malformed_key(mock_session, sample_dataset):
    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1'],
        labelColumn='label',
        taskType=TaskType.CLASSIFICATION,
        algorithmId='logistic_regression',
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_job(mock_session, body, idempotency_key='bad key')

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail['error']['code'] == 'VALIDATION_ERROR'


@pytest.mark.asyncio
async def test_create_job_replays_cached_response(mock_session, sample_dataset, monkeypatch):
    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1'],
        labelColumn='label',
        taskType=TaskType.CLASSIFICATION,
        algorithmId='logistic_regression',
    )
    cached = {'jobId': 'cached-job', 'status': 'queued'}
    lookup = AsyncMock(return_value=cached)
    monkeypatch.setattr('src.api.idempotency.lookup', lookup)

    response = await create_job(mock_session, body, idempotency_key='k1')

    assert response.status_code == 202
    assert lookup.await_count == 1
    # The cached path must not insert a second job.
    mock_session.execute.assert_not_awaited()
