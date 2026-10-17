"""Job discovery: filter parsing, sorting, and opaque cursor pagination.

Backs ``GET /api/v1/jobs`` (FEATURE-13). Every query parameter is validated
here so the endpoint can answer with a single ``400 VALIDATION_ERROR`` whose
``details.field`` names the offending parameter.

The cursor stays opaque: it encodes the sort key, the id of the last row, and
a fingerprint of the active filter/sort set, so a cursor issued for one filter
cannot silently paginate another one.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Select, and_, case, func, or_, select

from src.api.deps import raise_http
from src.api.jobs_serialization import JOB_STATUS_ORDER
from src.api.schemas import JobStatus, TaskType
from src.db.models import Job

__all__ = [
    'JobFilters',
    'build_query',
    'decode_cursor',
    'encode_cursor',
    'parse_filters',
    'sort_value_for',
]

MAX_QUERY_LENGTH = 200

#: Sortable fields and the direction used when the caller omits `:asc`/`:desc`.
SORT_FIELDS: dict[str, str] = {
    'createdAt': 'desc',
    'updatedAt': 'desc',
    'status': 'asc',
    'algorithmId': 'asc',
}

_STATUS_ORDINAL: dict[str, int] = {status: index for index, status in enumerate(JOB_STATUS_ORDER)}

_STATUS_CASE = case({value: index for value, index in _STATUS_ORDINAL.items()}, value=Job.status)


@dataclass(frozen=True)
class JobFilters:
    """The validated filter and sort set of a jobs-list request."""

    statuses: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    algorithm_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    q: str | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None
    sort_field: str = 'createdAt'
    sort_direction: str = 'desc'

    @property
    def ascending(self) -> bool:
        """Whether the sort runs from low to high."""
        return self.sort_direction == 'asc'

    @property
    def sort_key(self) -> str:
        """The canonical ``field:direction`` representation of the sort."""
        return f'{self.sort_field}:{self.sort_direction}'


def _invalid(field: str, message: str, **extra: Any) -> None:
    """Raise the skl validation error for a query parameter."""
    details: dict[str, Any] = {'field': field}
    details.update(extra)
    raise_http(400, 'VALIDATION_ERROR', message, details=details)


def _split(value: str | None) -> tuple[str, ...]:
    """Split a comma-separated parameter into its non-empty items."""
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(',') if item.strip())


def _parse_timestamp(field: str, value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp query parameter."""
    if value is None or not value.strip():
        return None
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    except ValueError:
        _invalid(field, f'{field} must be an ISO-8601 timestamp')
        return None  # pragma: no cover - _invalid always raises
    return parsed


def _parse_sort(value: str | None) -> tuple[str, str]:
    """Parse the ``sort`` parameter into a (field, direction) pair."""
    if value is None or not value.strip():
        return 'createdAt', 'desc'
    raw = value.strip()
    field, separator, direction = raw.partition(':')
    field = field.strip()
    if field not in SORT_FIELDS:
        _invalid('sort', f'Unknown sort field: {field}', allowed=list(SORT_FIELDS))
    if not separator:
        return field, SORT_FIELDS[field]
    direction = direction.strip().lower()
    if direction not in ('asc', 'desc'):
        _invalid('sort', 'sort direction must be "asc" or "desc"')
    return field, direction


def parse_filters(
    *,
    statuses: str | None = None,
    task_types: str | None = None,
    algorithm_ids: str | None = None,
    tags: str | None = None,
    q: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    sort: str | None = None,
) -> JobFilters:
    """Validate the query parameters of ``GET /api/v1/jobs``."""
    status_values = _split(statuses)
    for status in status_values:
        if status not in {member.value for member in JobStatus}:
            _invalid('status', f'Unknown job status: {status}')

    task_values = _split(task_types)
    for task_type in task_values:
        if task_type not in {member.value for member in TaskType}:
            _invalid('taskType', f'Unknown task type: {task_type}')

    algorithm_values = _split(algorithm_ids)
    tag_values = tuple(tag.lower() for tag in _split(tags))

    search = q.strip() if q else None
    if search:
        if len(search) > MAX_QUERY_LENGTH:
            _invalid('q', f'q must be at most {MAX_QUERY_LENGTH} characters')
    else:
        search = None

    from_dt = _parse_timestamp('createdFrom', created_from)
    to_dt = _parse_timestamp('createdTo', created_to)
    if from_dt is not None and to_dt is not None and to_dt < from_dt:
        _invalid('createdTo', 'createdTo must not be earlier than createdFrom')

    sort_field, sort_direction = _parse_sort(sort)

    return JobFilters(
        statuses=status_values,
        task_types=task_values,
        algorithm_ids=algorithm_values,
        tags=tag_values,
        q=search,
        created_from=from_dt,
        created_to=to_dt,
        sort_field=sort_field,
        sort_direction=sort_direction,
    )


def fingerprint(filters: JobFilters) -> str:
    """Return a stable fingerprint of the active filter and sort set."""
    payload = json.dumps(
        {
            'algorithmId': list(filters.algorithm_ids),
            'createdFrom': filters.created_from.isoformat() if filters.created_from else '',
            'createdTo': filters.created_to.isoformat() if filters.created_to else '',
            'q': filters.q or '',
            'sort': filters.sort_key,
            'status': list(filters.statuses),
            'tag': list(filters.tags),
            'taskType': list(filters.task_types),
        },
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def sort_value_for(filters: JobFilters, job: Job) -> Any:
    """Return the JSON-serialisable sort key of ``job``."""
    if filters.sort_field == 'createdAt':
        return job.created_at.isoformat()
    if filters.sort_field == 'updatedAt':
        return job.updated_at.isoformat()
    if filters.sort_field == 'status':
        return job.status
    return job.algorithm_id


def _comparable_sort_value(filters: JobFilters, raw: Any) -> Any:
    """Convert a decoded cursor value into a value comparable with the column."""
    if filters.sort_field in ('createdAt', 'updatedAt'):
        if not isinstance(raw, str):
            raise_http(400, 'VALIDATION_ERROR', 'Invalid cursor', details={'field': 'cursor'})
        try:
            return datetime.fromisoformat(raw.replace('Z', '+00:00'))
        except ValueError:
            raise_http(400, 'VALIDATION_ERROR', 'Invalid cursor', details={'field': 'cursor'})
    if filters.sort_field == 'status':
        if not isinstance(raw, str) or raw not in _STATUS_ORDINAL:
            raise_http(400, 'VALIDATION_ERROR', 'Invalid cursor', details={'field': 'cursor'})
        return _STATUS_ORDINAL[raw]
    return raw


def encode_cursor(filters: JobFilters, sort_value: Any, job_id: uuid.UUID) -> str:
    """Encode an opaque pagination cursor."""
    payload = {
        'v': 1,
        'fp': fingerprint(filters),
        'k': sort_value,
        'id': str(job_id),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip('=')


def decode_cursor(cursor: str, filters: JobFilters) -> tuple[Any, uuid.UUID]:
    """Decode a cursor, rejecting one issued for a different filter/sort set."""
    try:
        padded = cursor + '=' * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
    except ValueError, binascii.Error, UnicodeDecodeError, TypeError:
        raise_http(400, 'VALIDATION_ERROR', 'Invalid cursor', details={'field': 'cursor'})
    if not isinstance(payload, dict) or payload.get('v') != 1:
        raise_http(400, 'VALIDATION_ERROR', 'Invalid cursor', details={'field': 'cursor'})
    if payload.get('fp') != fingerprint(filters):
        raise_http(
            400,
            'VALIDATION_ERROR',
            'Cursor does not match the active filters; restart pagination',
            details={'field': 'cursor'},
        )
    try:
        job_id = uuid.UUID(str(payload['id']))
    except KeyError, ValueError:
        raise_http(400, 'VALIDATION_ERROR', 'Invalid cursor', details={'field': 'cursor'})
    return _comparable_sort_value(filters, payload.get('k')), job_id


def _sort_expression(filters: JobFilters):
    """Return the SQL expression the rows are sorted by."""
    if filters.sort_field == 'createdAt':
        return Job.created_at
    if filters.sort_field == 'updatedAt':
        return Job.updated_at
    if filters.sort_field == 'status':
        return _STATUS_CASE
    return Job.algorithm_id


def build_query(filters: JobFilters, cursor: tuple[Any, uuid.UUID] | None = None) -> Select:
    """Build the ``SELECT`` for the jobs list with filters, sort, and keyset page."""
    stmt = select(Job)

    if filters.statuses:
        stmt = stmt.where(Job.status.in_(filters.statuses))
    if filters.task_types:
        stmt = stmt.where(Job.task_type.in_(filters.task_types))
    if filters.algorithm_ids:
        stmt = stmt.where(Job.algorithm_id.in_(filters.algorithm_ids))
    if filters.tags:
        stmt = stmt.where(Job.tags.contains(list(filters.tags)))
    if filters.q:
        pattern = f'%{filters.q.lower()}%'
        stmt = stmt.where(
            or_(
                func.lower(Job.name).like(pattern),
                func.lower(func.array_to_string(Job.tags, ' ')).like(pattern),
                func.lower(Job.algorithm_id).like(pattern),
            )
        )
    if filters.created_from is not None:
        stmt = stmt.where(Job.created_at >= filters.created_from)
    if filters.created_to is not None:
        stmt = stmt.where(Job.created_at < filters.created_to)

    key = _sort_expression(filters)
    if cursor is not None:
        last_value, last_id = cursor
        if filters.ascending:
            stmt = stmt.where(or_(key > last_value, and_(key == last_value, Job.id < last_id)))
        else:
            stmt = stmt.where(or_(key < last_value, and_(key == last_value, Job.id < last_id)))

    if filters.ascending:
        stmt = stmt.order_by(key.asc(), Job.id.desc())
    else:
        stmt = stmt.order_by(key.desc(), Job.id.desc())

    return stmt


async def queue_positions(session, job_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    """Return the number of queued jobs ahead of each queued job.

    The position is the rank of the job among the queued jobs ordered by
    creation time, so a job at the front of the queue reports ``0``.
    """
    queued_ids = list(job_ids)
    if not queued_ids:
        return {}
    ranked = (
        select(
            Job.id.label('id'),
            (func.row_number().over(order_by=(Job.created_at.asc(), Job.id.asc())) - 1).label(
                'ahead'
            ),
        )
        .where(Job.status == 'queued')
        .subquery()
    )
    result = await session.execute(
        select(ranked.c.id, ranked.c.ahead).where(ranked.c.id.in_(queued_ids))
    )
    return {row.id: int(row.ahead) for row in result}
