"""Job serialization utilities."""

from src.api.schemas import (
    ArtefactResponse,
    JobAcceptedResponse,
    JobResponse,
    JobStatus,
    TaskType,
)
from src.db.models import Artefact, Job, JobMetric

# Lifecycle order used by the poll payload / UI stepper.
JOB_STATUS_ORDER = [
    'queued',
    'validating',
    'training',
    'evaluating',
    'exporting',
    'retrying',
    'completed',
    'failed',
]


def _artefact_response(artefact: Artefact) -> ArtefactResponse:
    return ArtefactResponse(
        artefactId=artefact.id,
        format=artefact.format,
        filename=artefact.filename,
        byteSize=artefact.byte_size,
    )


def _job_accepted_response(job: Job, queue_position: int | None = None) -> JobAcceptedResponse:
    """Build the minimal job response for lists and 202 responses."""
    return JobAcceptedResponse(
        jobId=job.id,
        status=JobStatus(job.status),
        datasetId=job.dataset_id,
        taskType=TaskType(job.task_type),
        algorithmId=job.algorithm_id,
        createdAt=job.created_at,
        name=job.name,
        tags=list(job.tags or []),
        attempts=job.attempts,
        maxAttempts=job.max_attempts,
        nextAttemptAt=job.next_attempt_at,
        queuePosition=queue_position,
    )


def _job_detail_response(
    job: Job,
    artefacts: list[Artefact],
    metric: JobMetric | None,
    queue_position: int | None = None,
) -> JobResponse:
    """Build the full job poll payload with its artefacts and latest metric.

    Artefacts are passed explicitly (already queried in the async session) so
    the response never triggers a lazy load on ``job.artefacts``.
    """
    return JobResponse(
        jobId=job.id,
        status=JobStatus(job.status),
        datasetId=job.dataset_id,
        taskType=TaskType(job.task_type),
        algorithmId=job.algorithm_id,
        createdAt=job.created_at,
        updatedAt=job.updated_at,
        featureColumns=job.feature_columns,
        labelColumn=job.label_column,
        rlColumns=job.rl_columns,
        hyperparameters=job.hyperparameters,
        metrics=metric.metrics if metric else None,
        charts=metric.charts if metric else None,
        artefacts=[_artefact_response(a) for a in artefacts],
        libraryVersions=job.library_versions,
        error=({'code': job.error_code, 'message': job.error_message} if job.error_code else None),
        name=job.name,
        tags=list(job.tags or []),
        attempts=job.attempts,
        maxAttempts=job.max_attempts,
        nextAttemptAt=job.next_attempt_at,
        queuePosition=queue_position,
    )
