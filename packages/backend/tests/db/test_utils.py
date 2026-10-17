"""Unit tests for src/db/utils.py."""

from datetime import UTC, datetime

from src.db.utils import utc_now


def test_utc_now_returns_datetime():
    """Test that utc_now returns a datetime object."""
    result = utc_now()
    assert isinstance(result, datetime)


def test_utc_now_returns_utc():
    """Test that utc_now returns a datetime with UTC timezone."""
    result = utc_now()
    assert result.tzinfo == UTC


def test_utc_now_is_recent():
    """Test that utc_now returns a recent datetime."""
    before = datetime.now(UTC)
    result = utc_now()
    after = datetime.now(UTC)
    assert before <= result <= after
