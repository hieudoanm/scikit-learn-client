"""Request ID / correlation ID middleware.

Assigns a correlation id per request, honouring an inbound ``X-Request-Id``
header. The id is stored in the request context so error envelopes carry a
``requestId``, and echoed back on the response as ``X-Request-Id``.
"""

from __future__ import annotations

from uuid import uuid4

from starlette.requests import Request
from starlette.responses import Response

from src.api.request_context import set_request_id


async def request_id_middleware(request: Request, call_next) -> Response:
    """Assign a correlation id per request, honouring an inbound ``X-Request-Id``."""
    request_id = request.headers.get('X-Request-Id') or str(uuid4())
    set_request_id(request_id)
    response = await call_next(request)
    response.headers['X-Request-Id'] = request_id
    return response


def setup_request_id(application) -> None:
    """Register the request ID middleware on ``application``."""
    application.middleware('http')(request_id_middleware)
