"""SQLAlchemy database models for skl."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    desc,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .utils import utc_now


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
    }


class Dataset(Base):
    """Stored uploaded CSV dataset metadata."""

    __tablename__ = 'datasets'

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    columns: Mapped[list[dict]] = mapped_column(JSONB, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class Job(Base):
    """Training job record."""

    __tablename__ = 'jobs'

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False
    )
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    algorithm_id: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_columns: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    label_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    rl_columns: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    hyperparameters: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})
    split_ratio: Mapped[float] = mapped_column(nullable=False, default=0.2)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False, default=42)
    export_pkl: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    export_joblib: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    library_versions: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    artefacts: Mapped[list[Artefact]] = relationship(
        back_populates='job', cascade='all, delete-orphan'
    )

    __table_args__ = (
        CheckConstraint(
            status.in_(
                [
                    'queued',
                    'validating',
                    'training',
                    'evaluating',
                    'exporting',
                    'retrying',
                    'completed',
                    'failed',
                ]
            ),
            name='jobs_status_check',
        ),
        CheckConstraint(
            task_type.in_(
                [
                    'classification',
                    'regression',
                    'semi_supervised',
                    'clustering',
                    'anomaly_detection',
                    'reinforcement_learning',
                ]
            ),
            name='jobs_task_type_check',
        ),
        CheckConstraint(
            (
                task_type.in_(['classification', 'regression', 'semi_supervised'])
                & label_column.is_not(None)
            )
            | (
                task_type.in_(['clustering', 'anomaly_detection', 'reinforcement_learning'])
                & label_column.is_(None)
            ),
            name='jobs_label_column_matches_task',
        ),
        CheckConstraint(
            (task_type == 'reinforcement_learning') & rl_columns.is_not(None)
            | (task_type != 'reinforcement_learning') & rl_columns.is_(None),
            name='jobs_rl_columns_matches_task',
        ),
    )


# Discovery indexes (Phase 12). ``jobs_tags_idx`` uses GIN containment so a tag
# filter can require *all* requested tags (``tags @> ARRAY[...]``).
Index('jobs_created_at_id_idx', desc(Job.created_at), desc(Job.id))
Index('jobs_tags_idx', Job.tags, postgresql_using='gin')
Index('jobs_algorithm_id_idx', Job.algorithm_id)
Index('jobs_name_lower_idx', func.lower(Job.name))


class DatasetInsight(Base):
    """Per-column profile computed once at dataset upload time.

    Advisory only: nothing in this table blocks job creation.
    """

    __tablename__ = 'dataset_insights'

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey('datasets.id', ondelete='CASCADE'),
        primary_key=True,
    )
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_missing: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    columns: Mapped[list[dict]] = mapped_column(JSONB, nullable=False)
    warnings: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class JobMetric(Base):
    """Training metrics for a job."""

    __tablename__ = 'job_metrics'

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey('jobs.id', ondelete='CASCADE'),
        primary_key=True,
    )
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False)
    charts: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class Artefact(Base):
    """Training artefact file record."""

    __tablename__ = 'artefacts'

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False
    )
    job: Mapped[Job] = relationship(back_populates='artefacts')
    format: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    __table_args__ = (
        CheckConstraint(
            format.in_(['onnx', 'pkl', 'joblib']),
            name='artefacts_format_check',
        ),
    )


class IdempotencyKey(Base):
    """Idempotency key for job creation."""

    __tablename__ = 'idempotency_keys'

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey('jobs.id', ondelete='SET NULL'), nullable=True
    )
    response_status: Mapped[int] = mapped_column(Integer, nullable=False)
    response_body: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
