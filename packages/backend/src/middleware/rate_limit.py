"""Per-client rate-limiting middleware.

Tracks per-client request counts in a sliding time window and returns HTTP 429
when a client exceeds its allowance. The client key is the first address in
``X-Forwarded-For`` (when present) or the direct client host.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from fastapi import Request, Response
from starlette.responses import JSONResponse

from src.api.deps import build_error_body
from src.api.request_context import get_request_id
from src.config import Settings


class RateLimiter:
    """Sliding-window rate limiter keyed by client identifier."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, deque[float]] = {}

    def is_allowed(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)`` for ``key``."""
        now = now if now is not None else time.monotonic()
        window = self._windows.setdefault(key, deque())
        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= self.max_requests:
            retry_after = window[0] + self.window_seconds - now
            return False, max(retry_after, 0.0)
        window.append(now)
        return True, 0.0

    def reset(self) -> None:
        """Clear all windows (mainly for tests)."""
        self._windows.clear()


def client_key(request: Request) -> str:
    """Return the client identifier for ``request``."""
    forwarded = request.headers.get('x-forwarded-for')
    if forwarded:
        return forwarded.split(',')[0].strip()
    client = request.client
    return client.host if client else 'unknown'


def make_rate_limit_middleware(settings: Settings):
    """Build an HTTP middleware that enforces the configured rate limit.

    When disabled (``settings.rate_limit_enabled`` is false) the middleware is a
    no-op pass-through.
    """
    limiter = RateLimiter(settings.rate_limit_max_requests, settings.rate_limit_window_seconds)
    enabled = settings.rate_limit_enabled

    async def middleware(request: Request, call_next: Callable) -> Response:
        if not enabled:
            return await call_next(request)

        key = client_key(request)
        allowed, retry_after = limiter.is_allowed(key)

        if not allowed:
            body = build_error_body(
                code='RATE_LIMITED',
                message='Too many requests. Please slow down and retry later.',
                request_id=get_request_id(),
                details={'retryAfterSeconds': round(retry_after, 2)},
            )
            return JSONResponse(
                status_code=429,
                content=body,
                headers={'Retry-After': str(int(retry_after) + 1)},
            )

        return await call_next(request)

    return middleware


def setup_rate_limit(application, settings: Settings) -> None:
    """Register the rate-limit middleware on ``application`` using ``settings``."""
    application.middleware('http')(make_rate_limit_middleware(settings))
