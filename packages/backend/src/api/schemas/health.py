"""Health and readiness schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = 'ok'


class ReadyResponse(BaseModel):
    status: str = 'ok'


class WorkerStatusResponse(BaseModel):
    """Runtime worker counters for GET /health/worker."""

    maxConcurrentJobs: int
    runningJobs: int
    queuedJobs: int
    retryingJobs: int
    availableSlots: int
