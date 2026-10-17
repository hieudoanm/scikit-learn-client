"""HTTP middleware for the skl backend.

Groups all request/response middleware in one place:

- :mod:`src.middleware.cors` — cross-origin resource sharing
- :mod:`src.middleware.request_id` — per-request correlation id
- :mod:`src.middleware.timeout` — request deadline enforcement
- :mod:`src.middleware.compression` — response compression
- :mod:`src.middleware.rate_limit` — per-client rate limiting

Wire them all with :func:`setup_middleware`.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from src.api.rate_limit import RateLimiter, client_key
from src.middleware.compression import setup_compression
from src.middleware.cors import setup_cors
from src.middleware.rate_limit import setup_rate_limit
from src.middleware.request_id import setup_request_id
from src.middleware.setup import setup_middleware
from src.middleware.timeout import setup_timeout

__all__ = [
    'CORSMiddleware',
    'GZipMiddleware',
    'BaseHTTPMiddleware',
    'client_key',
    'RateLimiter',
    'setup_cors',
    'setup_request_id',
    'setup_timeout',
    'setup_compression',
    'setup_rate_limit',
    'setup_middleware',
]
