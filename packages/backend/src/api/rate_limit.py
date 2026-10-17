"""Rate-limit middleware (re-exported from :mod:`src.middleware.rate_limit`).

Kept as a compatibility shim so existing imports of ``src.api.rate_limit``
continue to work. Prefer importing from ``src.middleware.rate_limit``.
"""

from __future__ import annotations

from src.middleware.rate_limit import (  # noqa: F401
    RateLimiter,
    client_key,
    make_rate_limit_middleware,
)

__all__ = ['RateLimiter', 'client_key', 'make_rate_limit_middleware']
