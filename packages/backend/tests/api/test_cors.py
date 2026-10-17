"""Tests for the CORS configuration and the interactive documentation toggle.

Covers scenario 27: the allow-list comes from `SKL_CORS_ORIGINS`, a wildcard
origin is refused while credentials are enabled, and the interactive docs can
be switched off.
"""

import pytest
from fastapi.testclient import TestClient
from main import create_app
from src.config import Settings, get_settings, reset_settings


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
        'rate_limit_max_requests': 100,
        'rate_limit_window_seconds': 60,
        'compression_enabled': True,
        'compression_min_size': 500,
        'request_timeout_seconds': 30,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def client():
    """A test client for an app with the default CORS allow-list."""
    return TestClient(create_app(_settings()))


def test_the_default_allow_list_permits_localhost(client):
    """`http://localhost:3000` is allowed by default."""
    response = client.get('/healthz', headers={'Origin': 'http://localhost:3000'})
    assert response.headers.get('access-control-allow-origin') == 'http://localhost:3000'


def test_the_default_allow_list_refuses_another_origin(client):
    """`http://evil.example` is not allowed by default."""
    response = client.get('/healthz', headers={'Origin': 'http://evil.example'})
    assert 'access-control-allow-origin' not in response.headers


def test_a_custom_allow_list_refuses_another_origin():
    """With a custom allow-list, a different origin gets no CORS header."""
    client = TestClient(create_app(_settings(cors_origins=('https://app.example',))))

    refused = client.get('/healthz', headers={'Origin': 'http://evil.example'})
    assert 'access-control-allow-origin' not in refused.headers

    allowed = client.get('/healthz', headers={'Origin': 'https://app.example'})
    assert allowed.headers.get('access-control-allow-origin') == 'https://app.example'


def test_a_wildcard_origin_is_refused_with_credentials(monkeypatch):
    """`SKL_CORS_ORIGINS=*` is rejected while credentials are enabled."""
    monkeypatch.setenv('SKL_CORS_ORIGINS', '*')
    monkeypatch.setenv('SKL_CORS_ALLOW_CREDENTIALS', 'true')
    reset_settings()
    try:
        settings = get_settings()
        assert settings.cors_origins == ()

        client = TestClient(create_app(settings))
        response = client.get('/healthz', headers={'Origin': 'http://evil.example'})
        assert 'access-control-allow-origin' not in response.headers
    finally:
        reset_settings()


def test_the_preflight_request_is_answered():
    """A preflight request for an allowed origin carries the CORS headers."""
    client = TestClient(create_app(_settings(cors_origins=('http://localhost:3000',))))
    response = client.options(
        '/api/v1/jobs',
        headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
        },
    )
    assert response.status_code == 200
    assert response.headers.get('access-control-allow-origin') == 'http://localhost:3000'


def test_the_interactive_docs_can_be_switched_off():
    """`SKL_ENABLE_INTERACTIVE_DOCS=false` removes the schema UI."""
    client = TestClient(create_app(_settings(enable_interactive_docs=False)))
    assert client.get('/docs').status_code == 404
    assert client.get('/redoc').status_code == 404
    assert client.get('/openapi.json').status_code == 404


def test_the_interactive_docs_are_exposed_by_default():
    """By default the schema UI and the OpenAPI document are served."""
    client = TestClient(create_app(_settings()))
    assert client.get('/docs').status_code == 200
    assert client.get('/openapi.json').status_code == 200
