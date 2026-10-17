"""Health endpoints: /healthz and /readyz."""

from fastapi import APIRouter
from sqlalchemy import func, select, text

from src.api.deps import DBSession, raise_http
from src.api.schemas import HealthResponse, ReadyResponse, WorkerStatusResponse
from src.config import get_settings
from src.db.models import Job

router = APIRouter(tags=['health'])


@router.get('/healthz', summary='Liveness', response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Process liveness probe."""
    return HealthResponse()


@router.get('/readyz', summary='Readiness (PostgreSQL)', response_model=ReadyResponse)
async def readyz(session: DBSession) -> ReadyResponse:
    """Database reachability probe."""
    try:
        await session.execute(text('SELECT 1'))
    except Exception:  # noqa: BLE001
        raise_http(503, 'SERVICE_UNAVAILABLE', 'Database not reachable')
    return ReadyResponse()


@router.get('/health/worker', summary='Worker status', response_model=WorkerStatusResponse)
async def worker_status(session: DBSession) -> WorkerStatusResponse:
    """Report the worker queue against its configured concurrency limit.

    The counters come straight from the ``jobs`` table: ``runningJobs`` counts
    the non-terminal statuses, so a slot is only reported busy while an attempt
    is actually in flight. This makes "why is my job still queued?" answerable
    without reading the process logs.
    """
    settings = get_settings()
    running = (
        select(func.count())
        .select_from(Job)
        .where(Job.status.in_(('validating', 'training', 'evaluating', 'exporting')))
    )
    queued = select(func.count()).select_from(Job).where(Job.status == 'queued')
    retrying = select(func.count()).select_from(Job).where(Job.status == 'retrying')

    running_jobs = int((await session.execute(running)).scalar_one())
    queued_jobs = int((await session.execute(queued)).scalar_one())
    retrying_jobs = int((await session.execute(retrying)).scalar_one())

    return WorkerStatusResponse(
        maxConcurrentJobs=settings.max_concurrent_jobs,
        runningJobs=running_jobs,
        queuedJobs=queued_jobs,
        retryingJobs=retrying_jobs,
        availableSlots=max(0, settings.max_concurrent_jobs - running_jobs),
    )
