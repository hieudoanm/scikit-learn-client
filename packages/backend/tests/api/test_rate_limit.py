"""Tests for the rate-limit middleware.

Covers: per-client sliding-window limits, the 429 envelope with
``Retry-After``, ``X-Forwarded-For`` keying, and the enabled/disabled toggle.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from main import create_app
from src.config import Settings


def _settings(**overrides) -> Settings:
    """Build a settings object for the app factory."""
    defaults = {
        'database_url': 'postgresql+asyncpg://skl:skl@localhost:5432/skl',
        'data_dir': '/tmp/skl-test',  # noqa: S108 — a throwaway path for the app factory
        'max_upload_mb': 10,
        'log_level': 'INFO',
        'cors_origins': ('http://localhost:3000',),
        'cors_allow_credentials': True,
        'enable_interactive_docs': True,
        'max_concurrent_jobs': 1,
        'job_timeout_seconds': 1800,
        'job_max_attempts': 2,
        'job_retry_base_seconds': 2,
        'worker_stale_seconds': 3600,
        'rate_limit_enabled': True,
        'rate_limit_max_requests': 3,
        'rate_limit_window_seconds': 60,
        'compression_enabled': True,
        'compression_min_size': 500,
        'request_timeout_seconds': 30,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def client():
    """A test client with a small rate limit (3 requests per window)."""
    return TestClient(create_app(_settings()))


def test_requests_up_to_the_limit_succeed(client):
    """The first ``rate_limit_max_requests`` requests in a window are allowed."""
    for _ in range(3):
        response = client.get('/healthz')
        assert response.status_code == 200


def test_request_beyond_the_limit_is_rejected(client):
    """The request after the limit returns HTTP 429 with the skl envelope."""
    for _ in range(3):
        client.get('/healthz')

    response = client.get('/healthz')
    assert response.status_code == 429
    body = response.json()
    assert body['error']['code'] == 'RATE_LIMITED'
    assert 'retryAfterSeconds' in body['error']['details']
    assert 'retry-after' in response.headers


def test_retry_after_header_is_set(client):
    """A rejected response carries a ``Retry-After`` header."""
    for _ in range(3):
        client.get('/healthz')

    response = client.get('/healthz')
    assert response.status_code == 429
    retry_after = int(response.headers['retry-after'])
    assert retry_after >= 1


def test_rate_limit_can_be_disabled():
    """With ``rate_limit_enabled=false`` no request is rejected."""
    client = TestClient(create_app(_settings(rate_limit_enabled=False)))
    for _ in range(10):
        response = client.get('/healthz')
        assert response.status_code == 200


def test_different_clients_have_separate_limits():
    """Two distinct client keys each get their own allowance."""
    settings = _settings(rate_limit_max_requests=1)
    client = TestClient(create_app(settings))

    # Client A consumes its single allowance.
    response_a = client.get('/healthz', headers={'x-forwarded-for': '10.0.0.1'})
    assert response_a.status_code == 200
    refused_a = client.get('/healthz', headers={'x-forwarded-for': '10.0.0.1'})
    assert refused_a.status_code == 429

    # Client B still has its own allowance.
    response_b = client.get('/healthz', headers={'x-forwarded-for': '10.0.0.2'})
    assert response_b.status_code == 200


def test_falls_back_to_client_host_without_forwarded_for(client):
    """Without ``X-Forwarded-For`` the direct client host is the key."""
    for _ in range(3):
        client.get('/healthz')
    response = client.get('/healthz')
    assert response.status_code == 429


def test_request_id_header_is_present_on_429(client):
    """A rate-limited response still carries ``X-Request-Id``."""
    for _ in range(3):
        client.get('/healthz')

    response = client.get('/healthz')
    assert response.status_code == 429
    assert 'x-request-id' in response.headers
    assert response.headers['x-request-id']
