"""Pydantic schemas grouped by domain, re-exported for compatibility.

Import from here (``from src.api.schemas import JobResponse``) so the package
can be reorganised without touching every caller.
"""

from src.api.schemas.algorithms import AlgorithmResponse, AlgorithmsListResponse
from src.api.schemas.datasets import (
    ColumnInsight,
    ColumnPreview,
    DatasetInsightsResponse,
    DatasetPreview,
    DatasetResponse,
    DatasetWarning,
    NumericSummary,
    ValueCount,
)
from src.api.schemas.health import HealthResponse, ReadyResponse, WorkerStatusResponse
from src.api.schemas.jobs import (
    ArtefactResponse,
    ComparisonDataset,
    ComparisonJob,
    ComparisonWarning,
    CreateJobRequest,
    JobAcceptedResponse,
    JobComparisonResponse,
    JobErrorDetail,
    JobListResponse,
    JobResponse,
    RLColumns,
    UpdateJobMetadataRequest,
)
from src.api.schemas.shared import (
    ArtefactFormat,
    ErrorDetail,
    ErrorEnvelope,
    JobStatus,
    TaskType,
)

__all__ = [
    'AlgorithmResponse',
    'AlgorithmsListResponse',
    'ArtefactFormat',
    'ArtefactResponse',
    'ColumnInsight',
    'ColumnPreview',
    'ComparisonDataset',
    'ComparisonJob',
    'ComparisonWarning',
    'CreateJobRequest',
    'DatasetInsightsResponse',
    'DatasetPreview',
    'DatasetResponse',
    'DatasetWarning',
    'ErrorDetail',
    'ErrorEnvelope',
    'HealthResponse',
    'JobAcceptedResponse',
    'JobComparisonResponse',
    'JobErrorDetail',
    'JobListResponse',
    'JobResponse',
    'JobStatus',
    'NumericSummary',
    'ReadyResponse',
    'RLColumns',
    'TaskType',
    'UpdateJobMetadataRequest',
    'ValueCount',
    'WorkerStatusResponse',
]
