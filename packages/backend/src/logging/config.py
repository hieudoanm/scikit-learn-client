"""Logging configuration with structured formatting and multiple log levels."""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from src.api.request_context import get_request_id


class RequestIdFilter(logging.Filter):
    """Attach the request-scoped correlation id to every log record.

    Reads the ``request_context`` contextvar set by the request middleware, so
    log lines can be correlated with the ``requestId`` returned in error
    envelopes (``docs/08-tech-stack/backend/README.md`` rule 6).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or '-'
        return True


class TimestampFormatter(logging.Formatter):
    """Formatter that includes timestamp, level, request id, module, and message."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        request_id = getattr(record, 'request_id', '-')
        module = record.module
        level = record.levelname
        message = record.getMessage()
        return f'{timestamp} [{level}] [{request_id}] {module}: {message}'


def setup_logging(level: str = 'INFO') -> None:
    """Configure root logger with timestamped output."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(TimestampFormatter())
    handler.addFilter(RequestIdFilter())
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
