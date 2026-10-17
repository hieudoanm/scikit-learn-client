"""Add nullable charts JSONB column to job_metrics for result visualisation.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-19

This migration adds the `charts` column to the `job_metrics` table to store
bounded evaluation data for the Model report feature (confusion matrix,
precision/recall bars, scatter plots, etc.). The column is nullable and JSONB,
so jobs completed before this feature remain valid (charts is null).

Downgrade simply removes the column; chart data is disposable (it can be
regenerated from the job configuration and dataset if needed).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def _has_table(bind, table_name: str) -> bool:
    return table_name in sa.inspect(bind).get_table_names()


def _has_column(bind, table_name: str, column_name: str) -> bool:
    columns = sa.inspect(bind).get_columns(table_name)
    return any(col["name"] == column_name for col in columns)


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "job_metrics"):
        return

    if not _has_column(bind, "job_metrics", "charts"):
        op.add_column(
            "job_metrics",
            sa.Column("charts", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "job_metrics"):
        return

    if _has_column(bind, "job_metrics", "charts"):
        op.drop_column("job_metrics", "charts")
