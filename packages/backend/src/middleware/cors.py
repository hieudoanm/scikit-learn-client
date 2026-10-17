"""Cross-origin resource sharing middleware.

Wraps Starlette's :class:`CORSMiddleware` so the allow-list comes from the
central settings module. A wildcard origin is refused while credentials are
enabled (handled in :mod:`src.config`).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def setup_cors(application: FastAPI, settings) -> None:
    """Register CORS middleware on ``application`` using ``settings``."""
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=['*'],
        allow_headers=['*'],
    )
