"""Unit tests for the central configuration module (src/config.py).

Every environment variable the backend reads is resolved here, so the tests
cover the documented defaults, the parsing rules, and the graceful degradation
of an unusable value (warn once, fall back to the default).
"""

import pytest
from src.config import (
    DEFAULT_COMPRESSION_ENABLED,
    DEFAULT_COMPRESSION_MIN_SIZE,
    DEFAULT_CORS_ORIGINS,
    DEFAULT_DATA_DIR,
    DEFAULT_JOB_MAX_ATTEMPTS,
    DEFAULT_JOB_RETRY_BASE_SECONDS,
    DEFAULT_JOB_TIMEOUT_SECONDS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENT_JOBS,
    DEFAULT_MAX_UPLOAD_MB,
    DEFAULT_RATE_LIMIT_ENABLED,
    DEFAULT_RATE_LIMIT_MAX_REQUESTS,
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_WORKER_STALE_SECONDS,
    MAX_COMPRESSION_MIN_SIZE,
    MAX_CONCURRENT_JOBS,
    MAX_JOB_MAX_ATTEMPTS,
    MAX_JOB_RETRY_BASE_SECONDS,
    MAX_JOB_TIMEOUT_SECONDS,
    MAX_RATE_LIMIT_MAX_REQUESTS,
    MAX_RATE_LIMIT_WINDOW_SECONDS,
    MAX_REQUEST_TIMEOUT_SECONDS,
    MAX_UPLOAD_MB,
    MAX_WORKER_STALE_SECONDS,
    get_settings,
    reset_settings,
)

ENV_VARS = (
    'DATABASE_URL',
    'SKL_DATA_DIR',
    'SKL_MAX_UPLOAD_MB',
    'SKL_LOG_LEVEL',
    'SKL_CORS_ORIGINS',
    'SKL_CORS_ALLOW_CREDENTIALS',
    'SKL_ENABLE_INTERACTIVE_DOCS',
    'SKL_MAX_CONCURRENT_JOBS',
    'SKL_JOB_TIMEOUT_SECONDS',
    'SKL_JOB_MAX_ATTEMPTS',
    'SKL_JOB_RETRY_BASE_SECONDS',
    'SKL_WORKER_STALE_SECONDS',
    'SKL_RATE_LIMIT_ENABLED',
    'SKL_RATE_LIMIT_MAX_REQUESTS',
    'SKL_RATE_LIMIT_WINDOW_SECONDS',
    'SKL_COMPRESSION_ENABLED',
    'SKL_COMPRESSION_MIN_SIZE',
    'SKL_REQUEST_TIMEOUT_SECONDS',
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove every skl variable and drop the cached settings around a test."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    yield
    reset_settings()


def test_defaults():
    """With an empty environment every documented default applies."""
    settings = get_settings()
    assert settings.database_url == 'postgresql+asyncpg://skl:skl@localhost:5432/skl'
    assert str(settings.data_dir) == DEFAULT_DATA_DIR
    assert settings.max_upload_mb == DEFAULT_MAX_UPLOAD_MB
    assert settings.log_level == DEFAULT_LOG_LEVEL
    assert settings.cors_origins == DEFAULT_CORS_ORIGINS
    assert settings.cors_allow_credentials is True
    assert settings.enable_interactive_docs is True
    assert settings.max_concurrent_jobs == DEFAULT_MAX_CONCURRENT_JOBS
    assert settings.job_timeout_seconds == DEFAULT_JOB_TIMEOUT_SECONDS
    assert settings.job_max_attempts == DEFAULT_JOB_MAX_ATTEMPTS
    assert settings.job_retry_base_seconds == DEFAULT_JOB_RETRY_BASE_SECONDS
    assert settings.worker_stale_seconds == DEFAULT_WORKER_STALE_SECONDS
    assert settings.rate_limit_enabled is DEFAULT_RATE_LIMIT_ENABLED
    assert settings.rate_limit_max_requests == DEFAULT_RATE_LIMIT_MAX_REQUESTS
    assert settings.rate_limit_window_seconds == DEFAULT_RATE_LIMIT_WINDOW_SECONDS
    assert settings.compression_enabled is DEFAULT_COMPRESSION_ENABLED
    assert settings.compression_min_size == DEFAULT_COMPRESSION_MIN_SIZE
    assert settings.request_timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_settings_are_cached_until_reset(monkeypatch):
    """The settings are read once; `reset_settings` re-reads the environment."""
    monkeypatch.setenv('SKL_LOG_LEVEL', 'DEBUG')
    assert get_settings().log_level == 'DEBUG'

    monkeypatch.setenv('SKL_LOG_LEVEL', 'ERROR')
    assert get_settings().log_level == 'DEBUG'

    reset_settings()
    assert get_settings().log_level == 'ERROR'


def test_database_url_scheme_normalisation(monkeypatch):
    """`postgres://` and `postgresql://` become the asyncpg scheme."""
    monkeypatch.setenv('DATABASE_URL', 'postgres://user:pass@host:5432/db')
    assert get_settings().database_url == 'postgresql+asyncpg://user:pass@host:5432/db'

    reset_settings()
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pass@host:5432/db')
    assert get_settings().database_url == 'postgresql+asyncpg://user:pass@host:5432/db'

    reset_settings()
    monkeypatch.setenv('DATABASE_URL', 'postgresql+asyncpg://user:pass@host:5432/db')
    assert get_settings().database_url == 'postgresql+asyncpg://user:pass@host:5432/db'


def test_data_dir_must_be_absolute(monkeypatch):
    """A relative `SKL_DATA_DIR` falls back to the documented default."""
    monkeypatch.setenv('SKL_DATA_DIR', 'relative/path')
    assert str(get_settings().data_dir) == DEFAULT_DATA_DIR

    reset_settings()
    monkeypatch.setenv('SKL_DATA_DIR', '/srv/skl')
    assert str(get_settings().data_dir) == '/srv/skl'


def test_log_level_is_validated(monkeypatch):
    """An unknown log level falls back to INFO."""
    monkeypatch.setenv('SKL_LOG_LEVEL', 'verbose')
    assert get_settings().log_level == DEFAULT_LOG_LEVEL

    reset_settings()
    monkeypatch.setenv('SKL_LOG_LEVEL', 'debug')
    assert get_settings().log_level == 'DEBUG'


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('1', True),
        ('true', True),
        ('yes', True),
        ('on', True),
        ('0', False),
        ('false', False),
        ('no', False),
        ('off', False),
    ],
)
def test_boolean_parsing(monkeypatch, raw, expected):
    """Booleans accept the documented spellings."""
    monkeypatch.setenv('SKL_CORS_ALLOW_CREDENTIALS', raw)
    assert get_settings().cors_allow_credentials is expected


def test_boolean_falls_back_on_garbage(monkeypatch):
    """An unparsable boolean falls back to the default."""
    monkeypatch.setenv('SKL_ENABLE_INTERACTIVE_DOCS', 'maybe')
    assert get_settings().enable_interactive_docs is True


@pytest.mark.parametrize(
    ('name', 'attribute', 'maximum'),
    [
        ('SKL_MAX_UPLOAD_MB', 'max_upload_mb', MAX_UPLOAD_MB),
        ('SKL_MAX_CONCURRENT_JOBS', 'max_concurrent_jobs', MAX_CONCURRENT_JOBS),
        ('SKL_JOB_TIMEOUT_SECONDS', 'job_timeout_seconds', MAX_JOB_TIMEOUT_SECONDS),
        ('SKL_JOB_MAX_ATTEMPTS', 'job_max_attempts', MAX_JOB_MAX_ATTEMPTS),
        ('SKL_JOB_RETRY_BASE_SECONDS', 'job_retry_base_seconds', MAX_JOB_RETRY_BASE_SECONDS),
        ('SKL_WORKER_STALE_SECONDS', 'worker_stale_seconds', MAX_WORKER_STALE_SECONDS),
        ('SKL_RATE_LIMIT_MAX_REQUESTS', 'rate_limit_max_requests', MAX_RATE_LIMIT_MAX_REQUESTS),
        ('SKL_RATE_LIMIT_WINDOW_SECONDS', 'rate_limit_window_seconds',
         MAX_RATE_LIMIT_WINDOW_SECONDS),
        ('SKL_COMPRESSION_MIN_SIZE', 'compression_min_size', MAX_COMPRESSION_MIN_SIZE),
        ('SKL_REQUEST_TIMEOUT_SECONDS', 'request_timeout_seconds', MAX_REQUEST_TIMEOUT_SECONDS),
    ],
)
def test_bounded_integers(monkeypatch, name, attribute, maximum):
    """A bounded integer accepts its maximum and rejects anything above it."""
    monkeypatch.setenv(name, str(maximum))
    assert getattr(get_settings(), attribute) == maximum

    reset_settings()
    monkeypatch.setenv(name, str(maximum + 1))
    default = getattr(get_settings(), attribute)
    assert default != maximum + 1


def test_non_numeric_integer_falls_back(monkeypatch):
    """A non-numeric value falls back to the default."""
    monkeypatch.setenv('SKL_JOB_TIMEOUT_SECONDS', 'soon')
    assert get_settings().job_timeout_seconds == DEFAULT_JOB_TIMEOUT_SECONDS


def test_cors_origins_are_normalised(monkeypatch):
    """Origins are trimmed and lower-cased scheme/host; invalid entries drop."""
    monkeypatch.setenv(
        'SKL_CORS_ORIGINS',
        'http://localhost:3000, https://app.example ,ftp://bad.example,http://with/path',
    )
    settings = get_settings()
    assert settings.cors_origins == ('http://localhost:3000', 'https://app.example')


def test_cors_wildcard_rejected_with_credentials(monkeypatch):
    """A wildcard origin is refused while credentials are enabled."""
    monkeypatch.setenv('SKL_CORS_ORIGINS', '*')
    monkeypatch.setenv('SKL_CORS_ALLOW_CREDENTIALS', 'true')
    assert get_settings().cors_origins == ()


def test_cors_wildcard_allowed_without_credentials(monkeypatch):
    """A wildcard origin is allowed when credentials are disabled."""
    monkeypatch.setenv('SKL_CORS_ORIGINS', '*')
    monkeypatch.setenv('SKL_CORS_ALLOW_CREDENTIALS', 'false')
    assert get_settings().cors_origins == ('*',)


def test_max_upload_bytes_property():
    """`max_upload_bytes` is the megabyte value in bytes."""
    assert get_settings().max_upload_bytes == DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
