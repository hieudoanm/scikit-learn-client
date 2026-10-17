"""Job discovery endpoints: history, comparison, and multi-job bundles.

Lifecycle routes (create, poll, rename, delete, artefact download) live in
:mod:`src.api.jobs_lifecycle`, which owns ``_load_job`` and friends. This
module adds the read-only, many-job views on top.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from starlette.background import BackgroundTask

from src.api.deps import DBSession, raise_http
from src.api.ids import parse_id_list
from src.api.jobs_bundle import (
    SKIPPED_NO_ARTEFACTS,
    SKIPPED_NOT_COMPLETED,
    BundleJob,
    build_bundle,
    skipped_entry,
)
from src.api.jobs_comparison import compare_jobs, parse_ids
from src.api.jobs_discovery import (
    build_query,
    decode_cursor,
    encode_cursor,
    parse_filters,
    queue_positions,
    sort_value_for,
)
from src.api.jobs_lifecycle import _load_artefacts, _load_metric
from src.api.jobs_serialization import JOB_STATUS_ORDER, _job_accepted_response
from src.api.schemas import JobComparisonResponse, JobListResponse
from src.db.models import Job

router = APIRouter(tags=['jobs'])

__all__ = ['JOB_STATUS_ORDER', 'router']

# Maximum number of jobs a single multi-job artefact bundle may contain.
MAX_BUNDLE_IDS = 10

# Query parameters of GET /api/v1/jobs; a repeated parameter is rejected.
LIST_QUERY_PARAMS = (
    'limit',
    'cursor',
    'status',
    'taskType',
    'algorithmId',
    'tag',
    'q',
    'createdFrom',
    'createdTo',
    'sort',
)


def _reject_repeated(request: Request, *names: str) -> None:
    """Reject a repeated query parameter; the comma-separated form is required."""
    for name in names:
        if len(request.query_params.getlist(name)) > 1:
            raise_http(
                400,
                'VALIDATION_ERROR',
                f'Repeated query parameter: {name}; use the comma-separated form instead',
                details={'field': name},
            )


@router.get('/api/v1/jobs', summary='List training jobs', response_model=JobListResponse)
async def list_jobs(
    session: DBSession,
    request: Request,
    limit: int = 20,
    cursor: str | None = None,
    job_status: Annotated[str | None, Query(alias='status')] = None,
    task_type: Annotated[str | None, Query(alias='taskType')] = None,
    algorithm_id: Annotated[str | None, Query(alias='algorithmId')] = None,
    tag: str | None = None,
    q: str | None = None,
    created_from: Annotated[str | None, Query(alias='createdFrom')] = None,
    created_to: Annotated[str | None, Query(alias='createdTo')] = None,
    sort: str | None = None,
) -> JobListResponse:
    """List jobs with optional filters, sorting, and cursor pagination.

    Filters compose with AND. ``status``, ``taskType``, ``algorithmId``, and
    ``tag`` take comma-separated values; ``tag`` requires **all** the listed
    tags to be present.
    """
    _reject_repeated(request, *LIST_QUERY_PARAMS)
    if limit < 1 or limit > 100:
        raise_http(
            400,
            'VALIDATION_ERROR',
            'limit must be between 1 and 100',
            details={'field': 'limit'},
        )

    filters = parse_filters(
        statuses=job_status,
        task_types=task_type,
        algorithm_ids=algorithm_id,
        tags=tag,
        q=q,
        created_from=created_from,
        created_to=created_to,
        sort=sort,
    )

    decoded = decode_cursor(cursor, filters) if cursor is not None else None
    result = await session.execute(build_query(filters, decoded).limit(limit + 1))
    rows = list(result.scalars().all())

    has_more = len(rows) > limit
    jobs = rows[:limit]

    positions = await queue_positions(session, [job.id for job in jobs])

    next_cursor = None
    if has_more and jobs:
        last = jobs[-1]
        next_cursor = encode_cursor(filters, sort_value_for(filters, last), last.id)

    return JobListResponse(
        jobs=[_job_accepted_response(job, positions.get(job.id)) for job in jobs],
        nextCursor=next_cursor,
        hasMore=has_more,
    )


@router.get(
    '/api/v1/jobs/compare',
    summary='Compare 2-5 jobs',
    response_model=JobComparisonResponse,
)
async def compare(
    session: DBSession,
    ids: Annotated[str | None, Query()] = None,
) -> JobComparisonResponse:
    """Compare 2–5 jobs side by side, with warnings about unlike jobs."""
    job_ids = parse_ids(ids)
    return await compare_jobs(session, job_ids)


@router.get(
    '/api/v1/jobs/artefacts.zip',
    summary='Download the artefacts of several jobs as a ZIP bundle',
    response_class=FileResponse,
)
async def download_jobs_bundle(
    session: DBSession,
    ids: Annotated[str | None, Query()] = None,
) -> FileResponse:
    """Bundle the artefacts of 1–10 jobs into one ZIP archive."""
    job_ids = parse_id_list(ids, minimum=1, maximum=MAX_BUNDLE_IDS)

    result = await session.execute(select(Job).where(Job.id.in_(job_ids)))
    jobs = {job.id: job for job in result.scalars().all()}
    missing = [str(job_id) for job_id in job_ids if job_id not in jobs]
    if missing:
        raise_http(
            404,
            'JOB_NOT_FOUND',
            'One or more jobs were not found',
            details={'missingJobIds': missing},
        )

    skipped: list[dict] = []
    bundle_jobs: list[BundleJob] = []
    for job_id in job_ids:
        job = jobs[job_id]
        artefacts = await _load_artefacts(session, job_id)
        if not artefacts:
            reason = SKIPPED_NOT_COMPLETED if job.status != 'completed' else SKIPPED_NO_ARTEFACTS
            skipped.append(skipped_entry(job_id, reason))
            continue
        metric = await _load_metric(session, job_id)
        bundle_jobs.append(
            BundleJob(
                job=job,
                artefacts=artefacts,
                metrics=metric.metrics if metric else None,
            )
        )

    if not bundle_jobs:
        raise_http(
            404,
            'ARTEFACT_NOT_FOUND',
            'None of the selected jobs has artefacts',
            details={'jobIds': [str(job_id) for job_id in job_ids]},
        )

    bundle = build_bundle(bundle_jobs, skipped)
    headers = {}
    if bundle.missing_files:
        headers['X-skl-Missing-Files'] = ','.join(bundle.missing_files)
    return FileResponse(
        bundle.path,
        media_type='application/zip',
        filename='skl-jobs.zip',
        headers=headers,
        background=BackgroundTask(bundle.path.unlink, missing_ok=True),
    )
