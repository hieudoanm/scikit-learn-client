"""Background training worker for skl jobs.

Runs validation, fit, evaluation, and export on a worker thread pool, advancing
job status through the lifecycle: queued ->
 validating -> training -> evaluating
-> exporting -> completed/failed.

Phase 12 (FEATURE-17) adds three properties to the worker:

- **bounded concurrency** — at most ``SKL_MAX_CONCURRENT_JOBS`` attempts run
  at the same time; the other jobs stay ``queued``;
- **a per-attempt timeout** — an attempt that exceeds
  ``SKL_JOB_TIMEOUT_SECONDS`` fails with ``JOB_TIMEOUT`` and leaves no partial
  artefacts behind;
- **a retry budget** — transient failures are retried up to
  ``SKL_JOB_MAX_ATTEMPTS`` times with exponential backoff and full jitter;
  deterministic failures are never retried.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError

from src.config import get_settings
from src.db.models import Artefact, Job
from src.db.session import async_session
from src.db.utils import utc_now
from src.export.onnx import export_joblib, export_onnx, export_pickle
from src.logging.config import get_logger
from src.storage.filesystem import remove_job_files
from src.training.catalogue import RL_TASK_TYPE
from src.training.core.estimator import build_export_pipeline, library_versions
from src.training.core.pipeline import train_pipeline
from src.training.worker.store import (
    JobDeleted,
    JobFailure,
    _job_exists,
    _load_dataset_df,
    _record_artefact,
    _record_error,
    _record_metrics,
    _set_status,
)
from src.training.worker.validation import _validate_mapping

logger = get_logger('skl.worker')

# In-process job queue: job_id -> asyncio.Task
_job_tasks: dict[uuid.UUID, asyncio.Task] = {}

# Terminal statuses
TERMINAL = {'completed', 'failed'}

#: Statuses a job holds while the worker is working on it.
ACTIVE_STATUSES = ('validating', 'training', 'evaluating', 'exporting')

#: Upper bound of the retry backoff, in seconds.
MAX_BACKOFF_SECONDS = 30.0

#: Failure codes caused by the data, not by the environment.
DETERMINISTIC_CODES = frozenset(
    {
        'INVALID_COLUMN_MAP',
        'INVALID_LABEL_TYPE',
        'INSUFFICIENT_LABELS',
        'INVALID_HYPERPARAMETERS',
        'UNKNOWN_ALGORITHM',
        'TASK_ALGORITHM_MISMATCH',
    }
)

__all__ = [
    'JobDeleted',
    'JobFailure',
    'JobTimeout',
    'backoff_delay',
    'cancel_job',
    'enqueue_job',
    'is_terminal',
    'is_transient',
    'recover_interrupted_jobs',
]


class JobTimeout(Exception):
    """Raised when an attempt exceeds the configured wall-clock budget."""


# --- bounded concurrency -----------------------------------------------------

_semaphore: asyncio.Semaphore | None = None
_semaphore_limit = 0
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Return the semaphore that bounds the number of concurrent attempts."""
    global _semaphore, _semaphore_limit, _semaphore_loop
    limit = max(1, get_settings().max_concurrent_jobs)
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - defensive
        loop = None
    if _semaphore is None or _semaphore_limit != limit or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(limit)
        _semaphore_limit = limit
        _semaphore_loop = loop
    return _semaphore


def reset_worker_state() -> None:
    """Drop the cached semaphore and the task registry (used by the tests)."""
    global _semaphore, _semaphore_limit, _semaphore_loop
    _semaphore = None
    _semaphore_limit = 0
    _semaphore_loop = None
    _job_tasks.clear()


# --- retry helpers -----------------------------------------------------------


def backoff_delay(attempt: int, base_seconds: int) -> float:
    """Return the full-jitter backoff delay that follows a failed attempt.

    The delay is drawn uniformly from ``[0, base * 2^(attempt-1))`` seconds and
    is capped at 30 seconds. It is never negative.
    """
    exponent = max(0, attempt - 1)
    ceiling = min(MAX_BACKOFF_SECONDS, float(base_seconds) * (2**exponent))
    if ceiling <= 0:
        return 0.0
    # The jitter only spreads retries; it carries no security meaning.
    return random.random() * ceiling  # noqa: S311


def is_transient(exc: BaseException) -> bool:
    """Whether the failure was caused by the environment rather than by the data.

    A failure is transient when it is a database connection error, a disk or
    other OS-level error, or an unhandled interruption of the worker. Failures
    caused by the data — an invalid column map, an invalid label type,
    insufficient labels, invalid hyperparameters, or a ``ValueError`` /
    ``TypeError`` raised by the estimator — are deterministic and are never
    retried, because they would fail identically.
    """
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        return is_transient(cause)
    if isinstance(exc, JobDeleted):
        return False
    if isinstance(exc, JobFailure):
        message = str(exc)
        return not any(code in message for code in DETERMINISTIC_CODES)
    if isinstance(exc, (ValueError, TypeError, KeyError, FileNotFoundError)):
        return False
    if isinstance(exc, (OSError, DBAPIError, ConnectionError)):
        return True
    return False


def _failure_code(exc: BaseException) -> tuple[str, str]:
    """Map an exception to the job error ``code`` and ``message``."""
    if isinstance(exc, JobFailure):
        return exc.code, str(exc) or 'Job failed'
    message = str(exc) or 'Training failed'
    code = 'TRAINING_FAILED' if _is_validation_error(message) else 'INTERNAL_ERROR'
    return code, message


def _is_validation_error(message: str) -> bool:
    """Whether a worker error message is a known validation/fit failure code."""
    return any(
        marker in message
        for marker in ('INVALID_COLUMN_MAP', 'INVALID_LABEL_TYPE', 'INSUFFICIENT_LABELS')
    )


# --- persistence helpers -----------------------------------------------------


async def _record_failure(job_id: uuid.UUID, code: str, message: str) -> None:
    """Best-effort persistence of a job failure, isolated from the main flow."""
    try:
        async with async_session() as err_session:
            await _record_error(err_session, job_id, code, message)
    except Exception:  # noqa: BLE001
        logger.exception('Failed to record error for job %s', job_id)


async def _mark_attempt(job_id: uuid.UUID) -> None:
    """Record that an attempt has started (increments ``attempts``)."""
    try:
        async with async_session() as session:
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    attempts=Job.attempts + 1,
                    next_attempt_at=None,
                    updated_at=utc_now(),
                )
            )
            await session.commit()
    except Exception:  # noqa: BLE001
        logger.exception('Failed to record the attempt of job %s', job_id)


async def _schedule_retry(job_id: uuid.UUID, next_attempt_at: datetime) -> None:
    """Move the job to ``retrying`` and record when the next attempt is due."""
    try:
        async with async_session() as session:
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status='retrying',
                    next_attempt_at=next_attempt_at,
                    updated_at=utc_now(),
                )
            )
            await session.commit()
    except Exception:  # noqa: BLE001
        logger.exception('Failed to schedule the retry of job %s', job_id)


async def _set_status_safe(job_id: uuid.UUID, status: str) -> None:
    """Set the status of a job, ignoring a failure to do so."""
    try:
        async with async_session() as session:
            await _set_status(session, job_id, status)
    except Exception:  # noqa: BLE001
        logger.exception('Failed to set the status of job %s to %s', job_id, status)


async def _remove_artefacts(job_id: uuid.UUID) -> None:
    """Delete the artefact rows and the artefact files of a job."""
    try:
        async with async_session() as session:
            result = await session.execute(select(Artefact).where(Artefact.job_id == job_id))
            for artefact in result.scalars().all():
                await session.delete(artefact)
            await session.commit()
        remove_job_files(job_id)
    except Exception:  # noqa: BLE001
        logger.exception('Failed to remove the artefacts of job %s', job_id)


# --- the attempt -------------------------------------------------------------


async def _run_job(job_id: uuid.UUID) -> None:
    """Execute a single attempt of a training job end-to-end.

    The blocking work (reading the CSV, fitting, exporting) runs on a worker
    thread so the event loop stays responsive and the attempt timeout can fire.
    Failures are raised to the caller; :func:`_process_job` decides whether the
    failure is worth another attempt.
    """
    async with async_session() as session:
        # Load job
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            logger.error('Job %s not found', job_id)
            return
        logger.info('Starting job %s (task=%s, algo=%s)', job_id, job.task_type, job.algorithm_id)

        # --- validating ---
        await _set_status(session, job_id, 'validating')
        df = await _load_dataset_df(session, job.dataset_id)
        _validate_mapping(df, job)
        logger.debug('Job %s validation completed', job_id)

        # --- training ---
        await _set_status(session, job_id, 'training')
        try:
            pipeline, metrics, charts = await asyncio.to_thread(train_pipeline, job, df)
        except JobDeleted:
            raise
        except Exception as exc:  # noqa: BLE001
            raise JobFailure('TRAINING_FAILED', str(exc) or 'Training failed') from exc
        if not await _job_exists(session, job_id):
            logger.info('Job %s was deleted during training; stopping', job_id)
            return
        await _record_metrics(session, job_id, metrics, charts)
        logger.debug('Job %s training completed, metrics recorded', job_id)

        # --- evaluating ---
        await _set_status(session, job_id, 'evaluating')
        logger.debug('Job %s evaluating', job_id)

        # --- exporting ---
        await _set_status(session, job_id, 'exporting')

        if not await _job_exists(session, job_id):
            logger.info('Job %s was deleted before export; stopping', job_id)
            return

        X = df[job.feature_columns]

        # Export failures are attributed to EXPORT_FAILED, distinct from
        # validation/fit failures (TRAINING_FAILED) so the UI can guide the
        # user appropriately (FEATURE-07).
        try:
            # Always export ONNX (required). Semi-supervised exports the fitted
            # base estimator; pkl/joblib keep the wrapper. RL exports the Q
            # network under the FEATURE-11 [state, action] contract.
            if job.task_type == RL_TASK_TYPE:
                from src.export.onnx import export_onnx_rl

                onnx_bytes, onnx_filename, onnx_key, onnx_size = await asyncio.to_thread(
                    export_onnx_rl, pipeline, X, pipeline.actions, job_id
                )
            else:
                onnx_pipeline = await asyncio.to_thread(build_export_pipeline, pipeline)
                onnx_bytes, onnx_filename, onnx_key, onnx_size = await asyncio.to_thread(
                    export_onnx, onnx_pipeline, X, job_id
                )
            await _record_artefact(session, job_id, 'onnx', onnx_filename, onnx_key, onnx_size)
            logger.info('Job %s exported artefacts (onnx)', job_id)

            # Optional exports
            if job.export_pkl:
                _, pkl_filename, pkl_key, pkl_size = await asyncio.to_thread(
                    export_pickle, pipeline, job_id
                )
                await _record_artefact(session, job_id, 'pkl', pkl_filename, pkl_key, pkl_size)
            if job.export_joblib:
                _, joblib_filename, joblib_key, joblib_size = await asyncio.to_thread(
                    export_joblib, pipeline, job_id
                )
                await _record_artefact(
                    session, job_id, 'joblib', joblib_filename, joblib_key, joblib_size
                )
        except JobDeleted:
            raise
        except Exception as exc:  # noqa: BLE001
            raise JobFailure('EXPORT_FAILED', str(exc) or 'Model export failed') from exc

        # --- completed ---
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status='completed',
                library_versions=library_versions(),
                updated_at=utc_now(),
            )
        )
        await session.commit()
        logger.info('Job %s completed', job_id)


async def _run_attempt(job_id: uuid.UUID) -> None:
    """Take a worker slot and run one attempt of the job under the timeout."""
    settings = get_settings()
    async with _get_semaphore():
        await _mark_attempt(job_id)
        try:
            await asyncio.wait_for(_run_job(job_id), timeout=settings.job_timeout_seconds)
        except TimeoutError as exc:
            raise JobTimeout from exc


async def _process_job(job_id: uuid.UUID) -> None:
    """Run a job with bounded concurrency, a timeout, and a retry budget."""
    settings = get_settings()
    max_attempts = max(1, settings.job_max_attempts)

    for attempt in range(1, max_attempts + 1):
        try:
            await _run_attempt(job_id)
        except JobTimeout:
            timeout = settings.job_timeout_seconds
            logger.error('Job %s exceeded the %s s time limit', job_id, timeout)
            await _remove_artefacts(job_id)
            await _record_failure(
                job_id,
                'JOB_TIMEOUT',
                f'Training exceeded the {timeout} s time limit and was stopped.',
            )
            return
        except JobDeleted:
            logger.info('Job %s was deleted; stopping', job_id)
            return
        except Exception as exc:  # noqa: BLE001
            code, message = _failure_code(exc)
            if not is_transient(exc):
                logger.error('Job %s failed (%s): %s', job_id, code, message)
                await _record_failure(job_id, code, message)
                return
            if attempt >= max_attempts:
                logger.error(
                    'Job %s exhausted its retry budget after %s attempt(s): %s',
                    job_id,
                    attempt,
                    message,
                )
                await _record_failure(job_id, 'JOB_RETRY_EXHAUSTED', message)
                return
            delay = backoff_delay(attempt, settings.job_retry_base_seconds)
            logger.warning(
                'Job %s attempt %s/%s failed transiently (%s); retrying in %.2fs',
                job_id,
                attempt,
                max_attempts,
                code,
                delay,
            )
            await _schedule_retry(job_id, utc_now() + timedelta(seconds=delay))
            await asyncio.sleep(delay)
            await _set_status_safe(job_id, 'queued')
        else:
            logger.info('Job %s completed after %s attempt(s)', job_id, attempt)
            return


async def recover_interrupted_jobs() -> list[uuid.UUID]:
    """Recover the jobs that were interrupted by a restart.

    Jobs waiting for a retry are reset to ``queued`` and returned so the caller
    can re-enqueue them; jobs abandoned in an active status for longer than
    ``SKL_WORKER_STALE_SECONDS`` are recorded ``failed`` with
    ``WORKER_INTERRUPTED``.
    """
    settings = get_settings()
    cutoff = utc_now() - timedelta(seconds=settings.worker_stale_seconds)
    requeued: list[uuid.UUID] = []

    async with async_session() as session:
        result = await session.execute(select(Job).where(Job.status == 'retrying'))
        for job in result.scalars().all():
            job.status = 'queued'
            job.next_attempt_at = None
            job.updated_at = utc_now()
            requeued.append(job.id)

        stale_result = await session.execute(
            select(Job).where(
                Job.status.in_(ACTIVE_STATUSES),
                Job.updated_at < cutoff,
            )
        )
        for job in stale_result.scalars().all():
            job.status = 'failed'
            job.error_code = 'WORKER_INTERRUPTED'
            job.error_message = 'The worker was interrupted before the job finished.'
            job.updated_at = utc_now()

        await session.commit()

    if requeued:
        logger.info('Requeued %s job(s) that were waiting for a retry', len(requeued))
    return requeued


# --- queue -------------------------------------------------------------------


def enqueue_job(job_id: uuid.UUID) -> None:
    """Enqueue a job for background processing (in-process task)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - defensive
        loop = asyncio.get_event_loop()
    task = loop.create_task(_process_job(job_id))
    _job_tasks[job_id] = task
    task.add_done_callback(lambda _t: _job_tasks.pop(job_id, None))


def cancel_job(job_id: uuid.UUID) -> None:
    """Cancel the in-process training task for ``job_id``, if one is registered.

    Best-effort: the worker stops at its next ``await`` and the delete path then
    relies on ``JobDeleted`` for the residual race.
    """
    task = _job_tasks.pop(job_id, None)
    if task is not None and not task.done():
        task.cancel()


def is_terminal(status: str) -> bool:
    return status in TERMINAL
