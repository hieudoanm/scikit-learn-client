"""Wire up all HTTP middleware on the FastAPI application.

Middleware runs in reverse order of registration (last registered runs first
on the request path). The intended order is:

1. CORS (outermost — must see every request/response)
2. Compression
3. Timeout
4. Rate limit
5. Request ID / correlation ID (innermost — sets X-Request-Id before anything else)

``setup_middleware`` registers them so the effective execution order matches
the list above.
"""

from __future__ import annotations

from fastapi import FastAPI

from src.config import Settings

from .compression import setup_compression
from .cors import setup_cors
from .rate_limit import setup_rate_limit
from .request_id import setup_request_id
from .timeout import setup_timeout


def setup_middleware(application: FastAPI, settings: Settings) -> None:
    """Register all HTTP middleware on ``application`` using ``settings``.

    Registration order matters: Starlette applies middleware like an onion,
    outermost first. Register CORS last so it wraps everything; register
    request ID first so it sits closest to the route handler.
    """
    setup_compression(application, settings)
    setup_timeout(application, settings)
    setup_rate_limit(application, settings)
    setup_request_id(application)
    setup_cors(application, settings)
