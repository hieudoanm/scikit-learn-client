"""Database persistence helpers for the background training worker.

Split out of :mod:`src.training.worker` to keep each module small. These
helpers own the job row lifecycle writes (status, error, metrics, artefacts)
and the deleted-job race handling.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Artefact, Dataset, Job, JobMetric
from src.db.utils import utc_now
from src.storage.filesystem import dataset_path


class JobDeleted(Exception):
    """Raised internally when the job row disappears mid-training.

    ``DELETE /api/v1/jobs/{jobId}`` removes the row while the in-process worker
    task may still be running. Cancellation is cooperative, so if the row is
    removed while the worker is between ``await`` points the next write raises a
    foreign-key error. This signal lets the worker stop cleanly instead of
    logging a spurious failure.
    """


class JobFailure(Exception):
    """A worker failure carrying the job-level error ``code`` to record.

    The worker's failure handler reads ``exc.code`` to store the code on the
    job's ``error_code`` column (FEATURE-07). Validation/fit errors map to
    ``TRAINING_FAILED`` and export errors to ``EXPORT_FAILED``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


async def _set_status(session: AsyncSession, job_id: uuid.UUID, status: str) -> None:
    await session.execute(
        update(Job).where(Job.id == job_id).values(status=status, updated_at=utc_now())
    )
    await session.commit()


async def _record_error(session: AsyncSession, job_id: uuid.UUID, code: str, message: str) -> None:
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(
            status='failed',
            error_code=code,
            error_message=message,
            updated_at=utc_now(),
        )
    )
    await session.commit()


async def _job_exists(session: AsyncSession, job_id: uuid.UUID) -> bool:
    """Return whether the job row is still present (it may have been deleted)."""
    result = await session.execute(select(Job.id).where(Job.id == job_id))
    return result.scalar_one_or_none() is not None


async def _commit_for_job(session: AsyncSession, job_id: uuid.UUID) -> None:
    """Commit pending writes, converting a deleted-job FK violation to ``JobDeleted``.

    Any other ``IntegrityError`` is re-raised unchanged.
    """
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        if not await _job_exists(session, job_id):
            raise JobDeleted from exc
        raise


async def _record_metrics(
    session: AsyncSession,
    job_id: uuid.UUID,
    metrics: dict[str, float],
    charts: dict[str, Any] | None = None,
) -> None:
    session.add(JobMetric(job_id=job_id, metrics=metrics, charts=charts))
    await _commit_for_job(session, job_id)


async def _record_artefact(
    session: AsyncSession,
    job_id: uuid.UUID,
    fmt: str,
    filename: str,
    storage_key: str,
    byte_size: int,
) -> None:
    session.add(
        Artefact(
            job_id=job_id,
            format=fmt,
            filename=filename,
            byte_size=byte_size,
            storage_key=storage_key,
        )
    )
    await _commit_for_job(session, job_id)


async def _load_dataset_df(session: AsyncSession, dataset_id: uuid.UUID) -> pd.DataFrame:
    """Load the CSV of a dataset.

    The file is read on a worker thread so the event loop stays responsive and
    the attempt timeout can fire while a large CSV is being parsed.
    """
    result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise FileNotFoundError(f'Dataset {dataset_id} not found')
    path = dataset_path(dataset.storage_key)
    return await asyncio.to_thread(pd.read_csv, path)
