"""Response compression middleware.

Wraps Starlette's :class:`GZipMiddleware` so responses above the configured
minimum size are gzip-compressed when the client advertises ``Accept-Encoding:
gzip``.
"""

from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware


def setup_compression(application: FastAPI, settings) -> None:
    """Register GZip compression middleware on ``application`` using ``settings``."""
    application.add_middleware(
        GZipMiddleware,
        minimum_size=settings.compression_min_size,
    )
