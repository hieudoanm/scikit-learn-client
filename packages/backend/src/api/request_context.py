"""Request-scoped correlation id.

The id is set once per request by the middleware in ``main.py`` and read by
``raise_http`` and the exception handlers so every error envelope carries a
``requestId``.
"""

from __future__ import annotations

from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar('request_id', default='')


def set_request_id(request_id: str) -> None:
    """Store the correlation id for the current request."""
    _request_id.set(request_id)


def get_request_id() -> str:
    """Return the correlation id for the current request (or ``''`` if unset)."""
    return _request_id.get()
