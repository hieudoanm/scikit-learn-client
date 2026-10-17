"""Job comparison: a side-by-side view of 2–5 jobs (FEATURE-14).

Read-only: it reads job rows, metrics, and artefacts, and never mutates a job.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from src.api.deps import raise_http
from src.api.ids import parse_id_list
from src.api.schemas import (
    ComparisonDataset,
    ComparisonJob,
    ComparisonWarning,
    JobComparisonResponse,
)
from src.db.models import Artefact, Dataset, Job, JobMetric

__all__ = ['MAX_IDS', 'MIN_IDS', 'compare_jobs', 'parse_ids']

MIN_IDS = 2
MAX_IDS = 5

WARNING_MESSAGES = {
    'DIFFERENT_TASKS': 'Compared jobs use different task types.',
    'DIFFERENT_DATASETS': 'Compared jobs use different datasets.',
    'DIFFERENT_FEATURES': 'Compared jobs use different feature column sets.',
    'INCOMPLETE_JOBS': 'At least one compared job has not finished.',
}


def parse_ids(raw: str | None) -> list[uuid.UUID]:
    """Parse the comma-separated ``ids`` query parameter of the comparison."""
    return parse_id_list(raw, minimum=MIN_IDS, maximum=MAX_IDS)


async def compare_jobs(session, ids: list[uuid.UUID]) -> JobComparisonResponse:
    """Build the comparison payload for the given job ids, in the given order."""
    result = await session.execute(select(Job).where(Job.id.in_(ids)))
    jobs = {job.id: job for job in result.scalars().all()}
    missing = [str(job_id) for job_id in ids if job_id not in jobs]
    if missing:
        raise_http(
            404,
            'JOB_NOT_FOUND',
            'One or more jobs were not found',
            details={'missingJobIds': missing},
        )

    dataset_ids = {job.dataset_id for job in jobs.values()}
    datasets_result = await session.execute(select(Dataset).where(Dataset.id.in_(dataset_ids)))
    datasets = {dataset.id: dataset for dataset in datasets_result.scalars().all()}

    metrics_result = await session.execute(select(JobMetric).where(JobMetric.job_id.in_(ids)))
    metrics = {metric.job_id: metric for metric in metrics_result.scalars().all()}

    artefacts_result = await session.execute(select(Artefact).where(Artefact.job_id.in_(ids)))
    artefacts: dict[uuid.UUID, list[Artefact]] = {job_id: [] for job_id in ids}
    for artefact in artefacts_result.scalars().all():
        artefacts.setdefault(artefact.job_id, []).append(artefact)

    entries: list[ComparisonJob] = []
    for job_id in ids:
        job = jobs[job_id]
        dataset = datasets.get(job.dataset_id)
        metric = metrics.get(job_id)
        entries.append(
            ComparisonJob(
                jobId=job.id,
                name=job.name,
                tags=list(job.tags or []),
                status=job.status,
                taskType=job.task_type,
                algorithmId=job.algorithm_id,
                createdAt=job.created_at,
                dataset=ComparisonDataset(
                    datasetId=job.dataset_id,
                    filename=dataset.original_filename if dataset else '',
                    rowCount=dataset.row_count if dataset else 0,
                    columnCount=len(dataset.columns) if dataset else 0,
                ),
                featureColumns=list(job.feature_columns or []),
                labelColumn=job.label_column,
                splitRatio=job.split_ratio,
                randomSeed=job.random_seed,
                hyperparameters=dict(job.hyperparameters or {}),
                metrics=metric.metrics if metric else None,
                artefactFormats=[artefact.format for artefact in artefacts.get(job_id, [])],
                error=(
                    {'code': job.error_code, 'message': job.error_message}
                    if job.error_code
                    else None
                ),
            )
        )

    warnings: list[ComparisonWarning] = []
    if len({job.task_type for job in jobs.values()}) > 1:
        warnings.append(
            ComparisonWarning(code='DIFFERENT_TASKS', message=WARNING_MESSAGES['DIFFERENT_TASKS'])
        )
    if len({job.dataset_id for job in jobs.values()}) > 1:
        warnings.append(
            ComparisonWarning(
                code='DIFFERENT_DATASETS', message=WARNING_MESSAGES['DIFFERENT_DATASETS']
            )
        )
    if len({frozenset(job.feature_columns or []) for job in jobs.values()}) > 1:
        warnings.append(
            ComparisonWarning(
                code='DIFFERENT_FEATURES', message=WARNING_MESSAGES['DIFFERENT_FEATURES']
            )
        )
    if any(job.status != 'completed' and job_id not in metrics for job_id, job in jobs.items()):
        warnings.append(
            ComparisonWarning(code='INCOMPLETE_JOBS', message=WARNING_MESSAGES['INCOMPLETE_JOBS'])
        )

    return JobComparisonResponse(jobs=entries, warnings=warnings)
