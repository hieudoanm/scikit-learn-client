"""Unit tests for the job discovery helpers (src/api/jobs_discovery.py).

Covers the filter parsing, the sort handling, and the opaque cursor of
``GET /api/v1/jobs`` (FEATURE-13, scenario 31).
"""

import uuid
from datetime import UTC, datetime

import pytest
from src.api.jobs_discovery import (
    SORT_FIELDS,
    build_query,
    decode_cursor,
    encode_cursor,
    fingerprint,
    parse_filters,
    sort_value_for,
)


def _filters(**kwargs):
    """Parse a filter set from keyword arguments."""
    return parse_filters(**kwargs)


def test_parse_filters_defaults():
    """Without parameters the filter set is the default (newest first)."""
    filters = _filters()
    assert filters.statuses == ()
    assert filters.task_types == ()
    assert filters.algorithm_ids == ()
    assert filters.tags == ()
    assert filters.q is None
    assert filters.sort_field == 'createdAt'
    assert filters.sort_direction == 'desc'
    assert filters.ascending is False
    assert filters.sort_key == 'createdAt:desc'


def test_parse_filters_splits_comma_separated_values():
    """`status`, `taskType`, `algorithmId`, and `tag` take comma-separated values."""
    filters = _filters(
        statuses='queued, failed',
        task_types='classification,regression',
        algorithm_ids='logistic_regression',
        tags='Baseline, Release',
    )
    assert filters.statuses == ('queued', 'failed')
    assert filters.task_types == ('classification', 'regression')
    assert filters.algorithm_ids == ('logistic_regression',)
    # Tags are lower-cased so the filter matches the stored tags.
    assert filters.tags == ('baseline', 'release')


def test_parse_filters_rejects_unknown_status():
    """An unknown status is a validation error naming the field."""
    with pytest.raises(Exception) as exc_info:
        _filters(statuses='queued,exploded')
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'status' in str(exc_info.value)


def test_parse_filters_rejects_unknown_task_type():
    """An unknown task type is a validation error."""
    with pytest.raises(Exception) as exc_info:
        _filters(task_types='classification,teleportation')
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'taskType' in str(exc_info.value)


def test_parse_filters_rejects_unknown_sort_field():
    """An unknown sort field is a validation error listing the allowed ones."""
    with pytest.raises(Exception) as exc_info:
        _filters(sort='colour:asc')
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'sort' in str(exc_info.value)
    for field in SORT_FIELDS:
        assert field in str(exc_info.value)


def test_parse_filters_rejects_unknown_sort_direction():
    """An unknown sort direction is a validation error."""
    with pytest.raises(Exception) as exc_info:
        _filters(sort='createdAt:sideways')
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_parse_filters_rejects_inverted_date_range():
    """`createdTo` before `createdFrom` is a validation error."""
    with pytest.raises(Exception) as exc_info:
        _filters(created_from='2024-02-01T00:00:00Z', created_to='2024-01-01T00:00:00Z')
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'createdTo' in str(exc_info.value)


def test_parse_filters_rejects_unparsable_timestamp():
    """A malformed timestamp is a validation error."""
    with pytest.raises(Exception) as exc_info:
        _filters(created_from='yesterday')
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_parse_filters_rejects_overlong_query():
    """A `q` longer than the documented maximum is a validation error."""
    with pytest.raises(Exception) as exc_info:
        _filters(q='x' * 201)
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_parse_filters_parses_timestamps():
    """ISO-8601 timestamps are parsed into aware datetimes."""
    filters = _filters(created_from='2024-01-01T00:00:00Z', created_to='2024-02-01T00:00:00Z')
    assert filters.created_from == datetime(2024, 1, 1, tzinfo=UTC)
    assert filters.created_to == datetime(2024, 2, 1, tzinfo=UTC)


def test_fingerprint_changes_with_the_filter_set():
    """A cursor issued for one filter set cannot be used with another."""
    base = _filters(statuses='queued')
    other = _filters(statuses='failed')
    assert fingerprint(base) != fingerprint(other)
    assert fingerprint(base) == fingerprint(_filters(statuses='queued'))


def test_cursor_round_trip():
    """A cursor encodes the sort value and the id of the last row."""
    filters = _filters(sort='algorithmId:asc')
    job_id = uuid.uuid4()
    cursor = encode_cursor(filters, 'logistic_regression', job_id)
    value, decoded_id = decode_cursor(cursor, filters)
    assert value == 'logistic_regression'
    assert decoded_id == job_id


def test_cursor_rejects_a_malformed_value():
    """A cursor that is not base64 JSON is a validation error."""
    filters = _filters()
    with pytest.raises(Exception) as exc_info:
        decode_cursor('not-a-cursor', filters)
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'Invalid cursor' in str(exc_info.value)


def test_cursor_rejects_a_foreign_filter_set():
    """A cursor issued for another filter set is refused."""
    cursor = encode_cursor(_filters(statuses='queued'), 'x', uuid.uuid4())
    with pytest.raises(Exception) as exc_info:
        decode_cursor(cursor, _filters(statuses='failed'))
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_cursor_rejects_a_foreign_sort():
    """A cursor issued for another sort order is refused."""
    cursor = encode_cursor(_filters(sort='createdAt:desc'), '2024-01-01T00:00:00', uuid.uuid4())
    with pytest.raises(Exception) as exc_info:
        decode_cursor(cursor, _filters(sort='createdAt:asc'))
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_cursor_rejects_an_unknown_version():
    """A cursor with an unknown payload version is refused."""
    filters = _filters()
    cursor = encode_cursor(filters, 'x', uuid.uuid4())
    tampered = cursor[:-2] + ('AA' if not cursor.endswith('AA') else 'BB')
    with pytest.raises(Exception) as exc_info:
        decode_cursor(tampered, filters)
    assert 'VALIDATION_ERROR' in str(exc_info.value)


class _Job:
    """Minimal stand-in for a job row."""

    def __init__(self, status='queued', algorithm_id='logistic_regression'):
        self.id = uuid.uuid4()
        self.status = status
        self.algorithm_id = algorithm_id
        self.created_at = datetime(2024, 1, 1, tzinfo=UTC)
        self.updated_at = datetime(2024, 1, 2, tzinfo=UTC)


def test_sort_value_for_created_at():
    """The `createdAt` sort key is the ISO-8601 creation timestamp."""
    job = _Job()
    filters = _filters(sort='createdAt:desc')
    assert sort_value_for(filters, job) == job.created_at.isoformat()


def test_sort_value_for_updated_at():
    """The `updatedAt` sort key is the ISO-8601 update timestamp."""
    job = _Job()
    filters = _filters(sort='updatedAt:asc')
    assert sort_value_for(filters, job) == job.updated_at.isoformat()


def test_sort_value_for_status_and_algorithm():
    """The `status` and `algorithmId` sort keys are the raw column values."""
    job = _Job(status='training', algorithm_id='random_forest')
    assert sort_value_for(_filters(sort='status:asc'), job) == 'training'
    assert sort_value_for(_filters(sort='algorithmId:asc'), job) == 'random_forest'


def test_build_query_applies_the_filters():
    """Every active filter contributes a `WHERE` clause."""
    filters = _filters(
        statuses='failed',
        task_types='classification',
        algorithm_ids='logistic_regression',
        tags='baseline',
        q='forest',
        created_from='2024-01-01T00:00:00Z',
        created_to='2024-02-01T00:00:00Z',
    )
    sql = str(build_query(filters))
    assert 'jobs.status IN' in sql
    assert 'jobs.task_type IN' in sql
    assert 'jobs.algorithm_id IN' in sql
    assert 'jobs.tags' in sql
    assert 'jobs.created_at >=' in sql
    assert 'jobs.created_at <' in sql
    assert 'ORDER BY' in sql


def test_build_query_without_filters_is_ordered():
    """Without filters the query is ordered by creation time, newest first."""
    sql = str(build_query(_filters()))
    assert 'ORDER BY jobs.created_at DESC' in sql


def test_build_query_ascending_sort():
    """An ascending sort flips the ordering."""
    sql = str(build_query(_filters(sort='algorithmId:asc')))
    assert 'ORDER BY jobs.algorithm_id ASC' in sql


def test_build_query_paginates_with_a_cursor():
    """A cursor adds the keyset predicate to the query."""
    filters = _filters(sort='createdAt:desc')
    cursor = decode_cursor(
        encode_cursor(filters, '2024-01-01T00:00:00+00:00', uuid.uuid4()), filters
    )
    sql = str(build_query(filters, cursor))
    assert 'jobs.created_at <' in sql
