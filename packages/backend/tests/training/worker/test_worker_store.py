"""Unit tests for src/training/worker/store.py persistence helpers."""

import uuid
from unittest.mock import AsyncMock

import pytest
import src.training.worker.store as store
from sqlalchemy.exc import IntegrityError
from src.training.worker.store import JobDeleted


@pytest.mark.asyncio
async def test_commit_for_job_raises_job_deleted(monkeypatch):
    """A foreign-key failure on a missing job becomes JobDeleted, not a failure."""
    job_id = uuid.uuid4()
    session = AsyncMock()
    session.commit.side_effect = IntegrityError('INSERT', {}, Exception('fk'))
    monkeypatch.setattr(store, '_job_exists', AsyncMock(return_value=False))

    with pytest.raises(JobDeleted):
        await store._commit_for_job(session, job_id)
    session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_commit_for_job_reraises_when_job_still_exists(monkeypatch):
    """An unrelated integrity error is re-raised unchanged."""
    job_id = uuid.uuid4()
    session = AsyncMock()
    session.commit.side_effect = IntegrityError('INSERT', {}, Exception('other'))
    monkeypatch.setattr(store, '_job_exists', AsyncMock(return_value=True))

    with pytest.raises(IntegrityError):
        await store._commit_for_job(session, job_id)
