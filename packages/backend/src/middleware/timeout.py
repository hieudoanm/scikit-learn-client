"""Request timeout middleware.

Enforces a per-request deadline using :func:`asyncio.wait_for`. Requests that
exceed the configured timeout receive a 503 ``REQUEST_TIMEOUT`` error envelope.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from src.api.deps import build_error_body
from src.api.request_context import get_request_id


def setup_timeout(application: FastAPI, settings) -> None:
    """Register the timeout middleware on ``application`` using ``settings``.

    Wraps each request in ``asyncio.wait_for`` with
    ``settings.request_timeout_seconds`` as the deadline. A timeout produces a
    503 response with the skl error envelope.
    """
    timeout_seconds = settings.request_timeout_seconds

    @application.middleware('http')
    async def timeout_middleware(request: Request, call_next: Callable) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
        except TimeoutError:
            body = build_error_body(
                code='REQUEST_TIMEOUT',
                message='The request took too long to complete.',
                request_id=get_request_id(),
            )
            return JSONResponse(
                status_code=503,
                content=body,
            )
