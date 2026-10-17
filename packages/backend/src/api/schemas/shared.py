"""Shared enums and the error envelope.

The enums live here because more than one domain imports them; the error
envelope is the transport contract every router uses.
"""

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


class TaskType(StrEnum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    SEMI_SUPERVISED = 'semi_supervised'
    CLUSTERING = 'clustering'
    ANOMALY_DETECTION = 'anomaly_detection'
    REINFORCEMENT_LEARNING = 'reinforcement_learning'


class JobStatus(StrEnum):
    QUEUED = 'queued'
    VALIDATING = 'validating'
    TRAINING = 'training'
    EVALUATING = 'evaluating'
    EXPORTING = 'exporting'
    RETRYING = 'retrying'
    COMPLETED = 'completed'
    FAILED = 'failed'


class ArtefactFormat(StrEnum):
    ONNX = 'onnx'
    PKL = 'pkl'
    JOBLIB = 'joblib'


# ──────────────────────────────────────────────────────────────────────────────
# Error Envelope
# ──────────────────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None
    requestId: UUID


class ErrorEnvelope(BaseModel):
    error: ErrorDetail
