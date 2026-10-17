"""Normalise unhandled exceptions into a safe skl error envelope.

The API must never leak a driver string, a DSN, or a raw SQL fragment
(`FEATURE-07`). Every unhandled exception is mapped here to a stable code and
a generic, user-safe message before it reaches the response.
"""

from __future__ import annotations

from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError, SQLAlchemyError

# Message shown for a database failure; never the driver text.
_DATABASE_MESSAGE = 'A database error occurred. Please retry.'

# Message shown for anything else.
_INTERNAL_MESSAGE = 'Internal server error'


def classify_unhandled(exc: Exception) -> tuple[str, str]:
    """Return the ``(code, message)`` for an unhandled exception.

    Precedence matters: an :class:`IntegrityError` is also a
    :class:`DBAPIError`, so the most specific type is checked first. The
    message is always a fixed string — the raw ``str(exc)`` is logged by the
    caller, never returned.
    """
    if isinstance(exc, IntegrityError):
        return 'CONFLICT', 'The request conflicts with the current state.'
    if isinstance(exc, (OperationalError, DBAPIError, SQLAlchemyError)):
        return 'DATABASE_ERROR', _DATABASE_MESSAGE
    return 'INTERNAL_ERROR', _INTERNAL_MESSAGE
