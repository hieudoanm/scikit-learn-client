"""API dependencies for skl."""

from typing import Annotated

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.request_context import get_request_id
from src.db.session import get_session

DBSession = Annotated[AsyncSession, Depends(get_session)]


def build_error_body(
    code: str,
    message: str,
    request_id: str,
    details: dict | None = None,
) -> dict:
    """Build the flat skl error envelope ``{"error": {...}}``."""
    error: dict = {'code': code, 'message': message, 'requestId': request_id}
    if details is not None:
        error['details'] = details
    return {'error': error}


def raise_http(
    status_code: int,
    code: str,
    message: str,
    details: dict | None = None,
) -> None:
    """Raise an HTTPException carrying an skl error envelope.

    The envelope is attached to ``HTTPException.detail``; the custom handler in
    ``main.py`` unwraps it so the response body is the flat ``{"error": {...}}``
    rather than FastAPI's default ``{"detail": ...}`` wrapper.
    """
    body = build_error_body(code, message, get_request_id(), details)
    raise HTTPException(status_code=status_code, detail=body)
