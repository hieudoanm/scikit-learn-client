"""Job schemas: requests, responses, comparison, and metadata."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.api.schemas.shared import ArtefactFormat, JobStatus, TaskType

# ──────────────────────────────────────────────────────────────────────────────
# RL Columns
# ──────────────────────────────────────────────────────────────────────────────


class RLColumns(BaseModel):
    """Experience log columns for reinforcement_learning jobs."""

    actionColumn: str
    rewardColumn: str
    nextStateColumns: list[str] = Field(default_factory=list)
    doneColumn: str | None = None
    episodeColumn: str | None = None

    model_config = ConfigDict(extra='forbid')


# ──────────────────────────────────────────────────────────────────────────────
# Job Request/Response Models
# ──────────────────────────────────────────────────────────────────────────────


class CreateJobRequest(BaseModel):
    """Request body for POST /api/v1/jobs."""

    datasetId: UUID
    featureColumns: list[str] = Field(min_length=1)
    taskType: TaskType
    algorithmId: str
    labelColumn: str | None = None
    rlColumns: RLColumns | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    splitRatio: float = Field(default=0.2, gt=0, lt=1)
    randomSeed: int = Field(default=42, ge=0)
    exportPkl: bool = False
    exportJoblib: bool = False

    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def validate_label_rules(self) -> CreateJobRequest:
        """Enforce label/rlColumns rules per task type.

        Note: Full validation (dataset column existence, hyperparameter ranges,
        semi-supervised minimums) is performed in `validate_job_request()`.
        This validator only checks the structural rules.
        """
        label_required = {
            TaskType.CLASSIFICATION,
            TaskType.REGRESSION,
            TaskType.SEMI_SUPERVISED,
        }
        label_forbidden = {
            TaskType.CLUSTERING,
            TaskType.ANOMALY_DETECTION,
            TaskType.REINFORCEMENT_LEARNING,
        }

        if self.taskType in label_required:
            if not self.labelColumn:
                raise ValueError(f'labelColumn required for {self.taskType.value}')
        if self.taskType in label_forbidden:
            if self.labelColumn:
                raise ValueError(f'{self.taskType.value} must not set labelColumn')

        if self.taskType == TaskType.REINFORCEMENT_LEARNING:
            if not self.rlColumns:
                raise ValueError('rlColumns required for reinforcement_learning')
        elif self.rlColumns:
            raise ValueError('rlColumns only allowed for reinforcement_learning')

        return self


class JobAcceptedResponse(BaseModel):
    """Response for 202 Accepted from POST /api/v1/jobs."""

    jobId: UUID
    status: JobStatus
    datasetId: UUID
    taskType: TaskType
    algorithmId: str
    createdAt: datetime
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    attempts: int = 0
    maxAttempts: int = 2
    nextAttemptAt: datetime | None = None
    queuePosition: int | None = None


class ArtefactResponse(BaseModel):
    artefactId: UUID
    format: ArtefactFormat
    filename: str
    byteSize: int


class JobErrorDetail(BaseModel):
    code: str
    message: str


class JobResponse(BaseModel):
    """Full job detail from GET /api/v1/jobs/{jobId}."""

    jobId: UUID
    status: JobStatus
    datasetId: UUID
    taskType: TaskType
    algorithmId: str
    createdAt: datetime
    updatedAt: datetime
    featureColumns: list[str]
    labelColumn: str | None = None
    rlColumns: RLColumns | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int] | None = None
    charts: dict[str, Any] | None = None
    artefacts: list[ArtefactResponse] = Field(default_factory=list)
    libraryVersions: dict[str, str] | None = None
    error: JobErrorDetail | None = None
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    attempts: int = 0
    maxAttempts: int = 2
    nextAttemptAt: datetime | None = None
    queuePosition: int | None = None


class JobListResponse(BaseModel):
    jobs: list[JobAcceptedResponse]
    nextCursor: str | None = None
    hasMore: bool


# ──────────────────────────────────────────────────────────────────────────────
# Job metadata (PATCH /api/v1/jobs/{jobId})
# ──────────────────────────────────────────────────────────────────────────────


class UpdateJobMetadataRequest(BaseModel):
    """Request body for PATCH /api/v1/jobs/{jobId}.

    Only the human-readable metadata of a job is editable; the training
    configuration is immutable once the job is submitted.
    """

    name: str | None = Field(default=None, max_length=120)
    tags: list[str] = Field(default_factory=list, max_length=10)

    model_config = ConfigDict(extra='forbid')


# ──────────────────────────────────────────────────────────────────────────────
# Job comparison (GET /api/v1/jobs/compare)
# ──────────────────────────────────────────────────────────────────────────────


class ComparisonDataset(BaseModel):
    """Dataset summary attached to a compared job."""

    datasetId: UUID
    filename: str
    rowCount: int
    columnCount: int


class ComparisonJob(BaseModel):
    """One column of the comparison view."""

    jobId: UUID
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: JobStatus
    taskType: TaskType
    algorithmId: str
    createdAt: datetime
    dataset: ComparisonDataset
    featureColumns: list[str] = Field(default_factory=list)
    labelColumn: str | None = None
    splitRatio: float | None = None
    randomSeed: int | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int] | None = None
    artefactFormats: list[str] = Field(default_factory=list)
    error: JobErrorDetail | None = None


class ComparisonWarning(BaseModel):
    """A reason why the compared jobs are not strictly comparable."""

    code: str
    message: str


class JobComparisonResponse(BaseModel):
    """Response for GET /api/v1/jobs/compare."""

    jobs: list[ComparisonJob]
    warnings: list[ComparisonWarning] = Field(default_factory=list)
