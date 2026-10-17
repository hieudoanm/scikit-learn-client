"""Dataset schemas: upload metadata, column previews, and insights."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ColumnPreview(BaseModel):
    name: str
    inferredDtype: str
    nullCount: int = Field(ge=0)
    sampleValues: list[Any] = Field(default_factory=list)


class DatasetResponse(BaseModel):
    datasetId: UUID
    sha256: str
    filename: str
    byteSize: int
    columns: list[ColumnPreview]
    rowCount: int
    createdAt: datetime


# ──────────────────────────────────────────────────────────────────────────────
# Dataset insights (GET /api/v1/datasets/{datasetId}/insights)
# ──────────────────────────────────────────────────────────────────────────────


class NumericSummary(BaseModel):
    """Numeric summary of a column."""

    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None


class ValueCount(BaseModel):
    """One of the most frequent values of a column."""

    value: str
    count: int


class ColumnInsight(BaseModel):
    """Per-column profile."""

    name: str
    inferredType: str
    missingCount: int = 0
    missingRate: float = 0.0
    uniqueCount: int = 0
    numeric: NumericSummary | None = None
    topValues: list[ValueCount] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetWarning(BaseModel):
    """A dataset-level warning."""

    code: str
    column: str | None = None
    message: str


class DatasetPreview(BaseModel):
    """Raw preview rows, positionally aligned with ``columns``."""

    offset: int
    limit: int
    totalRows: int
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str | None]] = Field(default_factory=list)


class DatasetInsightsResponse(BaseModel):
    """Response for GET /api/v1/datasets/{datasetId}/insights."""

    datasetId: UUID
    sha256: str
    filename: str
    rowCount: int
    columnCount: int
    totalMissingCells: int = 0
    missingRate: float = 0.0
    columns: list[ColumnInsight] = Field(default_factory=list)
    warnings: list[DatasetWarning] = Field(default_factory=list)
    preview: DatasetPreview
