"""Job lifecycle endpoints: create, delete, metadata, and artefact downloads.

The discovery, comparison, and bundle endpoints live in their own modules; this
module owns the routes that move a job through its life (submit, rename,
delete) and the reads that belong to a single job.
"""

from __future__ import annotations

import re
import uuid
from typing import Annotated

from fastapi import APIRouter, Header, status
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy import delete, insert, select
from starlette.background import BackgroundTask

from src.api import idempotency
from src.api.deps import DBSession, raise_http
from src.api.jobs_bundle import BundleJob, build_bundle
from src.api.jobs_discovery import queue_positions
from src.api.jobs_metadata import normalise_name, normalise_tags
from src.api.jobs_serialization import _job_detail_response
from src.api.jobs_validation import validate_job_request
from src.api.schemas import (
    CreateJobRequest,
    JobAcceptedResponse,
    JobResponse,
    JobStatus,
    UpdateJobMetadataRequest,
)
from src.config import get_settings
from src.db.models import Artefact, Job, JobMetric
from src.db.utils import utc_now
from src.storage.filesystem import artefact_path, remove_job_files
from src.training.worker import cancel_job, enqueue_job

router = APIRouter(tags=['jobs'])

# `Idempotency-Key` charset/length, mirroring the OpenAPI parameter (docs/06 §8).
IDEMPOTENCY_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,255}$')


async def _load_job(session: DBSession, job_id: uuid.UUID) -> Job:
    """Load a job row or raise ``404 JOB_NOT_FOUND``."""
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise_http(404, 'JOB_NOT_FOUND', f'Job {job_id} not found')
    return job


async def _load_artefacts(session: DBSession, job_id: uuid.UUID) -> list[Artefact]:
    """Load the artefact rows of a job."""
    result = await session.execute(
        select(Artefact).where(Artefact.job_id == job_id).order_by(Artefact.created_at.asc())
    )
    return list(result.scalars().all())


async def _load_metric(session: DBSession, job_id: uuid.UUID) -> JobMetric | None:
    """Load the newest metrics row of a job."""
    result = await session.execute(
        select(JobMetric).where(JobMetric.job_id == job_id).order_by(JobMetric.created_at.desc())
    )
    return result.scalars().first()


@router.post(
    '/api/v1/jobs',
    summary='Enqueue training job',
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAcceptedResponse,
)
async def create_job(
    session: DBSession,
    body: CreateJobRequest,
    idempotency_key: Annotated[str | None, Header(alias='Idempotency-Key')] = None,
) -> JobAcceptedResponse | JSONResponse:
    """Validate and enqueue a training job. Returns 202 Accepted.

    When an ``Idempotency-Key`` header is supplied, the response is cached and
    replayed on retry (see :mod:`src.api.idempotency`).
    """
    if idempotency_key is not None and not IDEMPOTENCY_KEY_PATTERN.match(idempotency_key):
        raise_http(
            400,
            'VALIDATION_ERROR',
            'Idempotency-Key must be 1-255 characters of [a-zA-Z0-9_-]',
        )

    body_dict = body.model_dump(mode='json')
    body_hash = idempotency.request_hash(body_dict)

    if idempotency_key is not None:
        cached = await idempotency.lookup(session, idempotency_key, body_hash)
        if cached is not None:
            return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=cached)

    dataset_id, dataset, validated = await validate_job_request(session, body)

    # --- insert job ---
    job_id = uuid.uuid4()
    now = utc_now()
    settings = get_settings()
    stmt = insert(Job).values(
        id=job_id,
        dataset_id=dataset_id,
        task_type=body.taskType.value,
        algorithm_id=body.algorithmId,
        feature_columns=validated['feature_columns'],
        label_column=validated['label_column'],
        rl_columns=validated['rl_columns'],
        hyperparameters=validated['hyperparameters'],
        split_ratio=validated['split_ratio'],
        random_seed=validated['random_seed'],
        export_pkl=validated['export_pkl'],
        export_joblib=validated['export_joblib'],
        status='queued',
        max_attempts=settings.job_max_attempts,
        created_at=now,
        updated_at=now,
    )
    await session.execute(stmt)

    response = JobAcceptedResponse(
        jobId=job_id,
        status=JobStatus.QUEUED,
        datasetId=dataset_id,
        taskType=body.taskType,
        algorithmId=body.algorithmId,
        createdAt=now,
        maxAttempts=settings.job_max_attempts,
    )

    if idempotency_key is not None:
        await idempotency.record(
            session,
            idempotency_key,
            body_hash,
            job_id,
            response.model_dump(mode='json'),
        )

    await session.commit()

    # --- enqueue worker ---
    enqueue_job(job_id)

    return response


@router.get('/api/v1/jobs/{job_id}', summary='Poll job status', response_model=JobResponse)
async def get_job(session: DBSession, job_id: uuid.UUID) -> JobResponse:
    """Return the current state of a job."""
    job = await _load_job(session, job_id)

    artefacts = await _load_artefacts(session, job_id)
    metric = await _load_metric(session, job_id)
    positions = await queue_positions(session, [job.id]) if job.status == 'queued' else {}

    return _job_detail_response(job, artefacts, metric, positions.get(job.id))


@router.patch(
    '/api/v1/jobs/{job_id}',
    summary='Update job name and tags',
    response_model=JobResponse,
)
async def update_job_metadata(
    session: DBSession,
    job_id: uuid.UUID,
    body: UpdateJobMetadataRequest,
) -> JobResponse:
    """Update the human-readable metadata of a job.

    Only ``name`` and ``tags`` are editable; the training configuration of a
    job is immutable. A field that is absent from the request is left
    unchanged.
    """
    job = await _load_job(session, job_id)

    if 'name' in body.model_fields_set:
        job.name = normalise_name(body.name)
    if 'tags' in body.model_fields_set:
        job.tags = normalise_tags(body.tags)
    job.updated_at = utc_now()
    await session.commit()

    artefacts = await _load_artefacts(session, job_id)
    metric = await _load_metric(session, job_id)
    positions = await queue_positions(session, [job.id]) if job.status == 'queued' else {}

    return _job_detail_response(job, artefacts, metric, positions.get(job.id))


@router.delete(
    '/api/v1/jobs/{job_id}',
    summary='Delete job and files',
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_job(session: DBSession, job_id: uuid.UUID) -> Response:
    """Delete a job and its artefact files."""
    await _load_job(session, job_id)

    remove_job_files(job_id)
    cancel_job(job_id)

    await session.execute(delete(JobMetric).where(JobMetric.job_id == job_id))
    await session.execute(delete(Artefact).where(Artefact.job_id == job_id))
    await session.execute(delete(Job).where(Job.id == job_id))
    await session.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    '/api/v1/jobs/{job_id}/artefacts.zip',
    summary='Download the artefacts of one job as a ZIP bundle',
    response_class=FileResponse,
)
async def download_job_bundle(session: DBSession, job_id: uuid.UUID) -> FileResponse:
    """Bundle the artefacts of one job into a ZIP archive."""
    job = await _load_job(session, job_id)
    artefacts = await _load_artefacts(session, job_id)
    if not artefacts:
        raise_http(
            404,
            'ARTEFACT_NOT_FOUND',
            'Job has no artefacts',
            details={'jobId': str(job_id)},
        )

    metric = await _load_metric(session, job_id)
    bundle = build_bundle(
        [BundleJob(job=job, artefacts=artefacts, metrics=metric.metrics if metric else None)]
    )
    headers = {}
    if bundle.missing_files:
        headers['X-skl-Missing-Files'] = ','.join(bundle.missing_files)
    return FileResponse(
        bundle.path,
        media_type='application/zip',
        filename=f'{job_id}.zip',
        headers=headers,
        background=BackgroundTask(bundle.path.unlink, missing_ok=True),
    )


@router.get(
    '/api/v1/jobs/{job_id}/artefacts/{artefact_id}',
    summary='Download artefact',
)
async def download_artefact(
    session: DBSession, job_id: uuid.UUID, artefact_id: uuid.UUID
) -> FileResponse:
    """Stream an artefact file as an attachment."""
    result = await session.execute(
        select(Artefact).where(Artefact.id == artefact_id, Artefact.job_id == job_id)
    )
    artefact = result.scalar_one_or_none()
    if artefact is None:
        raise_http(404, 'ARTEFACT_NOT_FOUND', 'Artefact not found')

    job = await _load_job(session, job_id)
    if job.status != 'completed':
        raise_http(404, 'ARTEFACT_NOT_FOUND', 'Artefact not available')

    path = artefact_path(artefact.storage_key, job_id)
    if not path.exists():
        raise_http(404, 'ARTEFACT_NOT_FOUND', 'Artefact file missing')

    return FileResponse(
        path,
        media_type='application/octet-stream',
        filename=artefact.filename,
    )
