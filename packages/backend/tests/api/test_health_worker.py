"""Tests for the worker status endpoint (Phase 12 hardening)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.api import health
from src.config import Settings


def _settings(**overrides) -> Settings:
    """Build a settings object for the endpoint."""
    defaults = {
        'database_url': 'postgresql+asyncpg://skl:skl@localhost:5432/skl',
        'data_dir': '/tmp/skl-test',  # noqa: S108 — throwaway path for the call
        'max_upload_mb': 10,
        'log_level': 'INFO',
        'cors_origins': ('http://localhost:3000',),
        'cors_allow_credentials': True,
        'enable_interactive_docs': True,
        'max_concurrent_jobs': 3,
        'job_timeout_seconds': 1800,
        'job_max_attempts': 2,
        'job_retry_base_seconds': 5,
        'worker_stale_seconds': 300,
        'rate_limit_enabled': True,
        'rate_limit_max_requests': 100,
        'rate_limit_window_seconds': 60,
        'compression_enabled': True,
        'compression_min_size': 500,
        'request_timeout_seconds': 30,
    }
    return Settings(**{**defaults, **overrides})


@pytest.mark.asyncio
async def test_worker_status_reports_the_queue_against_the_limit(monkeypatch):
    """The counters come from the jobs table and honour the concurrency limit."""
    monkeypatch.setattr(health, 'get_settings', lambda: _settings())

    session = MagicMock()
    # running, queued, retrying — one scalar per execute() call.
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(scalar_one=MagicMock(return_value=2)),
            MagicMock(scalar_one=MagicMock(return_value=5)),
            MagicMock(scalar_one=MagicMock(return_value=1)),
        ]
    )

    response = await health.worker_status(session)

    assert response.maxConcurrentJobs == 3
    assert response.runningJobs == 2
    assert response.queuedJobs == 5
    assert response.retryingJobs == 1
    assert response.availableSlots == 1


@pytest.mark.asyncio
async def test_worker_status_never_reports_a_negative_slot(monkeypatch):
    """A running count above the limit clamps ``availableSlots`` to zero."""
    monkeypatch.setattr(health, 'get_settings', lambda: _settings(max_concurrent_jobs=1))

    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(scalar_one=MagicMock(return_value=4)),
            MagicMock(scalar_one=MagicMock(return_value=0)),
            MagicMock(scalar_one=MagicMock(return_value=0)),
        ]
    )

    response = await health.worker_status(session)

    assert response.availableSlots == 0
