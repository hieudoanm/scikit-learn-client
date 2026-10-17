"""Central configuration for the skl backend.

Every environment variable the backend reads is resolved here, exactly once, at
process start. Modules call :func:`get_settings` instead of touching
``os.environ`` directly, so the effective configuration lives in a single place
and an invalid value degrades to a documented default with one warning instead
of stopping the stack.

Contract: ``docs/11-deployment/environment-variables.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from src.logging.config import get_logger

logger = get_logger('skl.config')

__all__ = [
    'DEFAULT_CORS_ORIGINS',
    'Settings',
    'get_settings',
    'reset_settings',
]

# --- documented defaults ----------------------------------------------------

DEFAULT_DATABASE_URL = 'postgresql://skl:skl@localhost:5432/skl'
DEFAULT_DATA_DIR = '/var/lib/skl'
DEFAULT_MAX_UPLOAD_MB = 10
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_CORS_ORIGINS = ('http://localhost:3000', 'http://127.0.0.1:3000')
DEFAULT_CORS_ALLOW_CREDENTIALS = True
DEFAULT_ENABLE_INTERACTIVE_DOCS = True
DEFAULT_MAX_CONCURRENT_JOBS = 1
DEFAULT_JOB_TIMEOUT_SECONDS = 1800
DEFAULT_JOB_MAX_ATTEMPTS = 2
DEFAULT_JOB_RETRY_BASE_SECONDS = 2
DEFAULT_WORKER_STALE_SECONDS = 3600
DEFAULT_RATE_LIMIT_ENABLED = True
DEFAULT_RATE_LIMIT_MAX_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_COMPRESSION_ENABLED = True
DEFAULT_COMPRESSION_MIN_SIZE = 500
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30

# --- documented maxima ------------------------------------------------------

MAX_UPLOAD_MB = 1024
MAX_CONCURRENT_JOBS = 16
MAX_JOB_TIMEOUT_SECONDS = 86_400
MAX_JOB_MAX_ATTEMPTS = 5
MAX_JOB_RETRY_BASE_SECONDS = 60
MAX_WORKER_STALE_SECONDS = 86_400
MAX_RATE_LIMIT_MAX_REQUESTS = 10_000
MAX_RATE_LIMIT_WINDOW_SECONDS = 3600
MAX_COMPRESSION_MIN_SIZE = 65_536
MAX_REQUEST_TIMEOUT_SECONDS = 3600

LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR')

TRUE_VALUES = frozenset({'1', 'true', 'yes', 'on'})
FALSE_VALUES = frozenset({'0', 'false', 'no', 'off'})


@dataclass(frozen=True, slots=True)
class Settings:
    """Effective backend configuration."""

    database_url: str
    data_dir: Path
    max_upload_mb: int
    log_level: str
    cors_origins: tuple[str, ...]
    cors_allow_credentials: bool
    enable_interactive_docs: bool
    max_concurrent_jobs: int
    job_timeout_seconds: int
    job_max_attempts: int
    job_retry_base_seconds: int
    worker_stale_seconds: int
    rate_limit_enabled: bool
    rate_limit_max_requests: int
    rate_limit_window_seconds: int
    compression_enabled: bool
    compression_min_size: int
    request_timeout_seconds: int

    @property
    def max_upload_bytes(self) -> int:
        """Maximum accepted upload size in bytes."""
        return self.max_upload_mb * 1024 * 1024


# --- parsing helpers --------------------------------------------------------


def _warn(name: str, value: str, reason: str, fallback: object) -> None:
    """Log a single warning about an unusable environment value."""
    logger.warning(
        'Invalid value for %s=%r (%s); falling back to %r.', name, value, reason, fallback
    )


def _env(name: str) -> str | None:
    """Return a raw environment value, or ``None`` when unset/blank."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable."""
    raw = _env(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    _warn(name, raw, 'expected one of 1/0/true/false/yes/no/on/off', default)
    return default


def _int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Parse an integer environment variable bounded by ``minimum``/``maximum``."""
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        _warn(name, raw, 'not an integer', default)
        return default
    if value < minimum or value > maximum:
        _warn(name, raw, f'outside the allowed range {minimum}..{maximum}', default)
        return default
    return value


def _list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated list environment variable."""
    raw = _env(name)
    if raw is None:
        return default
    items = tuple(item.strip() for item in raw.split(',') if item.strip())
    if not items:
        return ()
    return items


def _normalise_origin(entry: str) -> str | None:
    """Return a canonical ``scheme://host[:port]`` origin, or ``None``.

    ``None`` means the entry is unusable: a trailing slash, a non-HTTP scheme,
    a missing host, or a path/query/fragment component.
    """
    if entry.endswith('/'):
        return None
    parts = urlsplit(entry)
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        return None
    if parts.path or parts.query or parts.fragment:
        return None
    if parts.username or parts.password:
        return None
    return f'{parts.scheme}://{parts.netloc}'


def _database_url(name: str = 'DATABASE_URL') -> str:
    """Resolve the SQLAlchemy async database URL."""
    raw = _env(name)
    if raw is None:
        raw = DEFAULT_DATABASE_URL
    url = raw
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql+asyncpg://', 1)
    elif url.startswith('postgresql://') and '+asyncpg' not in url:
        url = url.replace('postgresql://', 'postgresql+asyncpg://', 1)
    return url


def _data_dir() -> Path:
    """Resolve the artefact volume root."""
    raw = _env('SKL_DATA_DIR')
    if raw is None:
        return Path(DEFAULT_DATA_DIR)
    path = Path(raw)
    if not path.is_absolute():
        _warn('SKL_DATA_DIR', raw, 'must be an absolute path', DEFAULT_DATA_DIR)
        return Path(DEFAULT_DATA_DIR)
    return path


def _log_level() -> str:
    """Resolve the root log level."""
    raw = _env('SKL_LOG_LEVEL')
    if raw is None:
        return DEFAULT_LOG_LEVEL
    upper = raw.upper()
    if upper not in LOG_LEVELS:
        _warn('SKL_LOG_LEVEL', raw, f'expected one of {"/".join(LOG_LEVELS)}', DEFAULT_LOG_LEVEL)
        return DEFAULT_LOG_LEVEL
    return upper


def _cors_origins(allow_credentials: bool) -> tuple[str, ...]:
    """Resolve the CORS allow-list, dropping unusable entries."""
    raw_entries = _list('SKL_CORS_ORIGINS', DEFAULT_CORS_ORIGINS)
    origins: list[str] = []
    for entry in raw_entries:
        if entry == '*':
            if allow_credentials:
                _warn(
                    'SKL_CORS_ORIGINS',
                    entry,
                    'a wildcard origin is not allowed while credentials are enabled; '
                    'set SKL_CORS_ALLOW_CREDENTIALS=false to allow it',
                    'the wildcard was dropped',
                )
                continue
            origins.append('*')
            continue
        normalised = _normalise_origin(entry)
        if normalised is None:
            _warn('SKL_CORS_ORIGINS', entry, 'not a valid origin', 'the entry was dropped')
            continue
        origins.append(normalised)
    return tuple(origins)


def _load() -> Settings:
    """Build the settings from the current environment."""
    allow_credentials = _bool('SKL_CORS_ALLOW_CREDENTIALS', DEFAULT_CORS_ALLOW_CREDENTIALS)
    return Settings(
        database_url=_database_url(),
        data_dir=_data_dir(),
        max_upload_mb=_int(
            'SKL_MAX_UPLOAD_MB', DEFAULT_MAX_UPLOAD_MB, minimum=1, maximum=MAX_UPLOAD_MB
        ),
        log_level=_log_level(),
        cors_origins=_cors_origins(allow_credentials),
        cors_allow_credentials=allow_credentials,
        enable_interactive_docs=_bool(
            'SKL_ENABLE_INTERACTIVE_DOCS', DEFAULT_ENABLE_INTERACTIVE_DOCS
        ),
        max_concurrent_jobs=_int(
            'SKL_MAX_CONCURRENT_JOBS',
            DEFAULT_MAX_CONCURRENT_JOBS,
            minimum=1,
            maximum=MAX_CONCURRENT_JOBS,
        ),
        job_timeout_seconds=_int(
            'SKL_JOB_TIMEOUT_SECONDS',
            DEFAULT_JOB_TIMEOUT_SECONDS,
            minimum=1,
            maximum=MAX_JOB_TIMEOUT_SECONDS,
        ),
        job_max_attempts=_int(
            'SKL_JOB_MAX_ATTEMPTS',
            DEFAULT_JOB_MAX_ATTEMPTS,
            minimum=1,
            maximum=MAX_JOB_MAX_ATTEMPTS,
        ),
        job_retry_base_seconds=_int(
            'SKL_JOB_RETRY_BASE_SECONDS',
            DEFAULT_JOB_RETRY_BASE_SECONDS,
            minimum=1,
            maximum=MAX_JOB_RETRY_BASE_SECONDS,
        ),
        worker_stale_seconds=_int(
            'SKL_WORKER_STALE_SECONDS',
            DEFAULT_WORKER_STALE_SECONDS,
            minimum=1,
            maximum=MAX_WORKER_STALE_SECONDS,
        ),
        rate_limit_enabled=_bool('SKL_RATE_LIMIT_ENABLED', DEFAULT_RATE_LIMIT_ENABLED),
        rate_limit_max_requests=_int(
            'SKL_RATE_LIMIT_MAX_REQUESTS',
            DEFAULT_RATE_LIMIT_MAX_REQUESTS,
            minimum=1,
            maximum=MAX_RATE_LIMIT_MAX_REQUESTS,
        ),
        rate_limit_window_seconds=_int(
            'SKL_RATE_LIMIT_WINDOW_SECONDS',
            DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
            minimum=1,
            maximum=MAX_RATE_LIMIT_WINDOW_SECONDS,
        ),
        compression_enabled=_bool('SKL_COMPRESSION_ENABLED', DEFAULT_COMPRESSION_ENABLED),
        compression_min_size=_int(
            'SKL_COMPRESSION_MIN_SIZE',
            DEFAULT_COMPRESSION_MIN_SIZE,
            minimum=0,
            maximum=MAX_COMPRESSION_MIN_SIZE,
        ),
        request_timeout_seconds=_int(
            'SKL_REQUEST_TIMEOUT_SECONDS',
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
            minimum=1,
            maximum=MAX_REQUEST_TIMEOUT_SECONDS,
        ),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings, building them on first use."""
    global _settings
    if _settings is None:
        _settings = _load()
    return _settings


def reset_settings() -> None:
    """Drop the cached settings so the next call re-reads the environment.

    Used by the test suite; production code never calls it.
    """
    global _settings
    _settings = None
