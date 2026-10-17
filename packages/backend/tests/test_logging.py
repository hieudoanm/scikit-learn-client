"""Unit tests for the structured logging configuration (src/logging/config.py)."""

from __future__ import annotations

import logging

from src.api.request_context import set_request_id
from src.logging.config import RequestIdFilter, TimestampFormatter, get_logger, setup_logging


def _record(**kwargs) -> logging.LogRecord:
    defaults = {
        'name': 'skl.test',
        'level': logging.WARNING,
        'pathname': __file__,
        'lineno': 1,
        'msg': 'something %s',
        'args': ('happened',),
        'exc_info': None,
    }
    return logging.LogRecord(**{**defaults, **kwargs})


def test_setup_logging_installs_single_handler() -> None:
    setup_logging('DEBUG')
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.DEBUG
    # The handler carries the request-id filter so every line is correlatable.
    assert any(isinstance(f, RequestIdFilter) for f in root.handlers[0].filters)


def test_formatter_includes_timestamp_and_level() -> None:
    record = _record()
    line = TimestampFormatter().format(record)
    assert '[WARNING]' in line
    assert 'something happened' in line
    # ISO-8601 timestamp prefix.
    assert line.startswith('20')


def test_formatter_includes_request_id_from_context() -> None:
    set_request_id('req-123')
    record = _record()
    RequestIdFilter().filter(record)
    line = TimestampFormatter().format(record)
    assert '[req-123]' in line


def test_formatter_falls_back_when_no_request_id() -> None:
    set_request_id('')
    record = _record()
    RequestIdFilter().filter(record)
    line = TimestampFormatter().format(record)
    assert '[-]' in line


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger('skl.worker')
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'skl.worker'
