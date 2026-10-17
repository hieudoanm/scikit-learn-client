"""Tests for the unhandled-exception classifier (Phase 12 hardening).

The API must never leak a driver string, a DSN, or a raw SQL fragment. These
tests pin the mapping and the generic messages.
"""

import pytest
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError, SQLAlchemyError
from src.api.errors import classify_unhandled


@pytest.mark.parametrize(
    ('exc', 'code'),
    [
        (IntegrityError('stmt', {}, Exception('duplicate key value')), 'CONFLICT'),
        (OperationalError('stmt', {}, Exception('connection refused')), 'DATABASE_ERROR'),
        (DBAPIError('stmt', {}, Exception('boom')), 'DATABASE_ERROR'),
        (SQLAlchemyError('boom'), 'DATABASE_ERROR'),
        (ValueError('a raw value'), 'INTERNAL_ERROR'),
        (RuntimeError('a raw runtime error'), 'INTERNAL_ERROR'),
    ],
)
def test_classify_unhandled_maps_the_exception_type(exc, code):
    """Each exception family maps to a stable, documented code."""
    assert classify_unhandled(exc)[0] == code


def test_classify_unhandled_never_leaks_the_driver_message():
    """The returned message is generic regardless of the raw text."""
    secret = 'password=supersecret host=10.0.0.1 relation "jobs" does not exist'
    exc = DBAPIError('SELECT * FROM jobs', {}, Exception(secret))

    code, message = classify_unhandled(exc)

    assert code == 'DATABASE_ERROR'
    assert secret not in message
    assert message == 'A database error occurred. Please retry.'


def test_an_integrity_error_is_not_reported_as_a_plain_database_error():
    """``IntegrityError`` is the most specific type and wins the mapping."""
    exc = IntegrityError('INSERT', {}, Exception('unique constraint violated'))

    code, message = classify_unhandled(exc)

    assert code == 'CONFLICT'
    assert 'unique constraint' not in message
