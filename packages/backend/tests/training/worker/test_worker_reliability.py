"""Unit tests for the worker reliability features (FEATURE-17).

Covers the bounded concurrency, the per-attempt timeout, the retry budget with
full-jitter backoff, and the recovery of the jobs interrupted by a restart
(scenarios 28-30).
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import src.training.worker.worker as worker
from src.training.worker.store import JobFailure
from src.training.worker.worker import (
    JobTimeout,
    backoff_delay,
    is_transient,
    recover_interrupted_jobs,
    reset_worker_state,
)


class _FakeSessionCM:
    """Async context manager yielding a fake AsyncSession-like object."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


def _patch_worker(monkeypatch, job, *, dataset_error=None):
    """Patch the worker's collaborators so an attempt runs without a database."""
    session = AsyncMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=job))
    )

    recorded: dict = {}

    async def fake_record_error(_session, _job_id, code, message):
        recorded['code'] = code
        recorded['message'] = message

    monkeypatch.setattr(worker, 'async_session', lambda: _FakeSessionCM(session))
    monkeypatch.setattr(
        worker,
        '_load_dataset_df',
        AsyncMock(
            side_effect=dataset_error,
            return_value=pd.DataFrame({'feature1': [1.0, 2.0], 'label': [0, 1]}),
        ),
    )
    monkeypatch.setattr(worker, '_validate_mapping', MagicMock())
    monkeypatch.setattr(worker, '_set_status', AsyncMock())
    monkeypatch.setattr(worker, '_job_exists', AsyncMock(return_value=True))
    monkeypatch.setattr(worker, '_record_metrics', AsyncMock())
    monkeypatch.setattr(worker, '_record_artefact', AsyncMock())
    monkeypatch.setattr(worker, '_record_error', fake_record_error)
    monkeypatch.setattr(worker, '_remove_artefacts', AsyncMock())
    return recorded


def _job():
    """Build a job row for the worker."""
    job = MagicMock()
    job.id = uuid.uuid4()
    job.dataset_id = uuid.uuid4()
    job.task_type = 'classification'
    job.algorithm_id = 'logistic_regression'
    job.feature_columns = ['feature1']
    job.label_column = 'label'
    job.export_pkl = False
    job.export_joblib = False
    return job


@pytest.fixture(autouse=True)
def _reset_worker_state():
    """Drop the cached semaphore between the tests."""
    reset_worker_state()
    yield
    reset_worker_state()


# --- classification ----------------------------------------------------------


def test_is_transient_for_os_errors():
    """An OS-level error is transient."""
    assert is_transient(OSError('disk full')) is True


def test_is_transient_for_value_errors():
    """A `ValueError` raised by the estimator is deterministic."""
    assert is_transient(ValueError('bad input')) is False


def test_is_transient_for_type_errors():
    """A `TypeError` is deterministic."""
    assert is_transient(TypeError('bad type')) is False


def test_is_transient_for_job_failures():
    """A `JobFailure` inherits the classification of its cause."""
    try:
        try:
            raise OSError('disk full')
        except OSError as exc:
            raise JobFailure('EXPORT_FAILED', 'export failed') from exc
    except JobFailure as exc:
        assert is_transient(exc) is True

    try:
        try:
            raise ValueError('INVALID_COLUMN_MAP: nope')
        except ValueError as exc:
            raise JobFailure('TRAINING_FAILED', 'fit failed') from exc
    except JobFailure as exc:
        assert is_transient(exc) is False


def test_is_transient_for_unknown_exceptions():
    """An unrecognised exception is treated as deterministic."""
    assert is_transient(RuntimeError('boom')) is False


# --- backoff -----------------------------------------------------------------


def test_backoff_delay_is_within_the_documented_window():
    """The delay is drawn from `[0, base * 2^(n-1))`."""
    for attempt in range(1, 8):
        ceiling = min(30.0, 2.0 * (2 ** (attempt - 1)))
        for _ in range(20):
            delay = backoff_delay(attempt, 2)
            assert 0.0 <= delay < ceiling


def test_backoff_delay_is_capped_at_thirty_seconds():
    """The delay never exceeds the 30 second cap."""
    assert backoff_delay(20, 60) < 30.0


def test_backoff_delay_is_never_negative():
    """The delay is never negative."""
    assert backoff_delay(1, 2) >= 0.0


# --- timeout -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_timeout_is_recorded(monkeypatch):
    """An attempt that exceeds the time limit fails with JOB_TIMEOUT."""
    job = _job()
    recorded = _patch_worker(monkeypatch, job)

    async def _slow(*args, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(worker, '_run_job', _slow)
    monkeypatch.setattr(worker, 'get_settings', lambda: _settings(timeout=1))

    await worker._process_job(job.id)

    assert recorded['code'] == 'JOB_TIMEOUT'
    assert '1 s time limit' in recorded['message']


def _settings(**overrides):
    """Build a settings stub for the worker."""
    settings = MagicMock()
    settings.job_timeout_seconds = overrides.get('timeout', 1800)
    settings.job_max_attempts = overrides.get('max_attempts', 2)
    settings.job_retry_base_seconds = overrides.get('retry_base', 2)
    settings.max_concurrent_jobs = overrides.get('max_concurrent', 1)
    settings.worker_stale_seconds = overrides.get('stale', 3600)
    return settings


# --- retry -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transient_failure_is_retried_and_then_succeeds(monkeypatch):
    """A transient failure followed by a success completes with two attempts."""
    job = _job()
    recorded = _patch_worker(monkeypatch, job)

    attempts: list[int] = []

    async def _flaky(job_id):
        attempts.append(job_id)
        if len(attempts) == 1:
            raise OSError('temporary disk failure')
        return None

    monkeypatch.setattr(worker, '_run_job', _flaky)
    monkeypatch.setattr(worker, 'get_settings', lambda: _settings(max_attempts=2))
    monkeypatch.setattr(worker, 'backoff_delay', lambda attempt, base: 0.0)

    await worker._process_job(job.id)

    assert len(attempts) == 2
    assert 'code' not in recorded


@pytest.mark.asyncio
async def test_deterministic_failure_is_not_retried(monkeypatch):
    """A deterministic failure ends the job after exactly one attempt."""
    job = _job()
    recorded = _patch_worker(monkeypatch, job)

    attempts: list[int] = []

    async def _broken(job_id):
        attempts.append(job_id)
        raise ValueError('INVALID_COLUMN_MAP: label is also a feature')

    monkeypatch.setattr(worker, '_run_job', _broken)
    monkeypatch.setattr(worker, 'get_settings', lambda: _settings(max_attempts=3))

    await worker._process_job(job.id)

    assert len(attempts) == 1
    assert recorded['code'] == 'TRAINING_FAILED'


@pytest.mark.asyncio
async def test_exhausted_retry_budget(monkeypatch):
    """A job that keeps failing transiently ends with JOB_RETRY_EXHAUSTED."""
    job = _job()
    recorded = _patch_worker(monkeypatch, job)

    attempts: list[int] = []

    async def _broken(job_id):
        attempts.append(job_id)
        raise OSError('temporary disk failure')

    monkeypatch.setattr(worker, '_run_job', _broken)
    monkeypatch.setattr(worker, 'get_settings', lambda: _settings(max_attempts=2))
    monkeypatch.setattr(worker, 'backoff_delay', lambda attempt, base: 0.0)

    await worker._process_job(job.id)

    assert len(attempts) == 2
    assert recorded['code'] == 'JOB_RETRY_EXHAUSTED'


@pytest.mark.asyncio
async def test_retry_schedules_the_next_attempt(monkeypatch):
    """A retry moves the job to `retrying` and records the next attempt time."""
    job = _job()
    _patch_worker(monkeypatch, job)

    scheduled: list[str] = []

    async def _fake_schedule(job_id, next_attempt_at):
        scheduled.append(job_id)

    async def _broken(job_id):
        raise OSError('temporary disk failure')

    monkeypatch.setattr(worker, '_run_job', _broken)
    monkeypatch.setattr(worker, '_schedule_retry', _fake_schedule)
    monkeypatch.setattr(worker, 'get_settings', lambda: _settings(max_attempts=2))
    monkeypatch.setattr(worker, 'backoff_delay', lambda attempt, base: 0.0)

    await worker._process_job(job.id)

    assert scheduled == [job.id]


# --- concurrency -------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_is_bounded(monkeypatch):
    """At most `SKL_MAX_CONCURRENT_JOBS` attempts run at the same time."""
    running = 0
    peak = 0

    async def _slow(job_id):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1

    monkeypatch.setattr(worker, '_run_job', _slow)
    monkeypatch.setattr(worker, 'get_settings', lambda: _settings(max_concurrent=2))
    monkeypatch.setattr(worker, '_mark_attempt', AsyncMock())
    monkeypatch.setattr(worker, '_record_error', AsyncMock())

    await asyncio.gather(
        *(worker._process_job(uuid.uuid4()) for _ in range(6)),
    )

    assert peak <= 2


@pytest.mark.asyncio
async def test_a_queued_job_waits_for_a_free_slot(monkeypatch):
    """With one slot a second job only starts after the first one finished."""
    order: list[str] = []
    first = uuid.uuid4()
    second = uuid.uuid4()

    async def _slow(job_id):
        order.append(f'start:{job_id}')
        await asyncio.sleep(0.02)
        order.append(f'end:{job_id}')

    monkeypatch.setattr(worker, '_run_job', _slow)
    monkeypatch.setattr(worker, 'get_settings', lambda: _settings(max_concurrent=1))
    monkeypatch.setattr(worker, '_mark_attempt', AsyncMock())
    monkeypatch.setattr(worker, '_record_error', AsyncMock())

    await asyncio.gather(worker._process_job(first), worker._process_job(second))

    assert order[0] == f'start:{first}'
    assert order[1] == f'end:{first}'
    assert order[2] == f'start:{second}'


# --- recovery ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_interrupted_jobs_requeues_and_fails(monkeypatch):
    """Retrying jobs are requeued; stale active jobs are failed."""
    retrying = MagicMock()
    retrying.id = uuid.uuid4()
    retrying.status = 'retrying'
    retrying.next_attempt_at = None

    stale = MagicMock()
    stale.id = uuid.uuid4()
    stale.status = 'training'
    stale.error_code = None
    stale.error_message = None

    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[retrying])))
            ),
            MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[stale])))
            ),
        ]
    )
    monkeypatch.setattr(worker, 'async_session', lambda: _FakeSessionCM(session))
    monkeypatch.setattr(worker, 'get_settings', lambda: _settings(stale=1))

    requeued = await recover_interrupted_jobs()

    assert requeued == [retrying.id]
    assert retrying.status == 'queued'
    assert retrying.next_attempt_at is None
    assert stale.status == 'failed'
    assert stale.error_code == 'WORKER_INTERRUPTED'


@pytest.mark.asyncio
async def test_recover_interrupted_jobs_without_pending_jobs(monkeypatch):
    """Nothing to recover answers with an empty list."""
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))),
            MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))),
        ]
    )
    monkeypatch.setattr(worker, 'async_session', lambda: _FakeSessionCM(session))
    monkeypatch.setattr(worker, 'get_settings', lambda: _settings())

    assert await recover_interrupted_jobs() == []


def test_job_timeout_is_an_exception():
    """`JobTimeout` is the internal signal of a timed-out attempt."""
    assert issubclass(JobTimeout, Exception)
