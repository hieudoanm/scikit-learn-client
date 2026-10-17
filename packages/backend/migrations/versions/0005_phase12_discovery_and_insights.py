"""Add job discovery metadata, retry bookkeeping, and dataset insights.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-21

Phase 12 (`docs/10-implementation-plan/phase-12-hardening-and-insights.md`):

- widen the `jobs.status` CHECK to include the non-terminal `retrying` status;
- add `jobs.name` (optional human-readable label) and `jobs.tags` (`TEXT[]`);
- add the retry bookkeeping columns `attempts`, `max_attempts`, `next_attempt_at`;
- add the discovery indexes used by `GET /api/v1/jobs`
  (`jobs_created_at_id_idx`, `jobs_tags_idx` GIN, `jobs_algorithm_id_idx`,
  `jobs_name_lower_idx`);
- create the `dataset_insights` table holding the per-column profile computed
  at upload time.

Existing rows keep working: `name` is NULL, `tags` is `{}`, `attempts` is 0,
`max_attempts` is 2, and `next_attempt_at` is NULL.

`init_db()` builds a fresh database from the SQLAlchemy metadata, so this
revision is a no-op there; it upgrades existing volumes in place.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

_STATUS_OLD = (
    "'queued', 'validating', 'training', 'evaluating', 'exporting', 'completed', 'failed'"
)
_STATUS_NEW = (
    "'queued', 'validating', 'training', 'evaluating', 'exporting', "
    "'retrying', 'completed', 'failed'"
)


def _has_table(bind, table_name: str) -> bool:
    return table_name in sa.inspect(bind).get_table_names()


def _has_column(bind, table_name: str, column_name: str) -> bool:
    columns = sa.inspect(bind).get_columns(table_name)
    return any(col["name"] == column_name for col in columns)


def _has_constraint(bind, table_name: str, constraint_name: str) -> bool:
    constraints = sa.inspect(bind).get_check_constraints(table_name)
    return any(c["name"] == constraint_name for c in constraints)


def _has_index(bind, table_name: str, index_name: str) -> bool:
    indexes = sa.inspect(bind).get_indexes(table_name)
    return any(i["name"] == index_name for i in indexes)


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "jobs"):
        return

    if not _has_column(bind, "jobs", "name"):
        op.add_column("jobs", sa.Column("name", sa.String(length=120), nullable=True))
    if not _has_column(bind, "jobs", "tags"):
        op.add_column(
            "jobs",
            sa.Column(
                "tags",
                sa.dialects.postgresql.ARRAY(sa.Text()),
                nullable=False,
                server_default=sa.text("'{}'::text[]"),
            ),
        )
    if not _has_column(bind, "jobs", "attempts"):
        op.add_column(
            "jobs",
            sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        )
    if not _has_column(bind, "jobs", "max_attempts"):
        op.add_column(
            "jobs",
            sa.Column("max_attempts", sa.Integer(), nullable=False, server_default=sa.text("2")),
        )
    if not _has_column(bind, "jobs", "next_attempt_at"):
        op.add_column(
            "jobs",
            sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        )

    if _has_constraint(bind, "jobs", "jobs_status_check"):
        op.drop_constraint("jobs_status_check", "jobs", type_="check")
        op.create_check_constraint("jobs_status_check", "jobs", f"status IN ({_STATUS_NEW})")

    if not _has_index(bind, "jobs", "jobs_created_at_id_idx"):
        op.create_index(
            "jobs_created_at_id_idx", "jobs", [sa.text("created_at DESC"), sa.text("id DESC")]
        )
    if not _has_index(bind, "jobs", "jobs_tags_idx"):
        op.create_index(
            "jobs_tags_idx", "jobs", ["tags"], postgresql_using="gin"
        )
    if not _has_index(bind, "jobs", "jobs_algorithm_id_idx"):
        op.create_index("jobs_algorithm_id_idx", "jobs", ["algorithm_id"])
    if not _has_index(bind, "jobs", "jobs_name_lower_idx"):
        op.create_index("jobs_name_lower_idx", "jobs", [sa.text("lower(name)")])

    if not _has_table(bind, "dataset_insights"):
        op.create_table(
            "dataset_insights",
            sa.Column("dataset_id", sa.dialects.postgresql.UUID(), nullable=False),
            sa.Column("row_count", sa.Integer(), nullable=False),
            sa.Column("column_count", sa.Integer(), nullable=False),
            sa.Column(
                "total_missing",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column("columns", sa.dialects.postgresql.JSONB(), nullable=False),
            sa.Column(
                "warnings",
                sa.dialects.postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'[]'::jsonb"),
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.ForeignKeyConstraint(
                ["dataset_id"], ["datasets.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("dataset_id"),
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _has_table(bind, "dataset_insights"):
        op.drop_table("dataset_insights")

    if _has_table(bind, "jobs"):
        if _has_index(bind, "jobs", "jobs_name_lower_idx"):
            op.drop_index("jobs_name_lower_idx", table_name="jobs")
        if _has_index(bind, "jobs", "jobs_algorithm_id_idx"):
            op.drop_index("jobs_algorithm_id_idx", table_name="jobs")
        if _has_index(bind, "jobs", "jobs_tags_idx"):
            op.drop_index("jobs_tags_idx", table_name="jobs")
        if _has_index(bind, "jobs", "jobs_created_at_id_idx"):
            op.drop_index("jobs_created_at_id_idx", table_name="jobs")

        if _has_constraint(bind, "jobs", "jobs_status_check"):
            op.drop_constraint("jobs_status_check", "jobs", type_="check")
            op.create_check_constraint("jobs_status_check", "jobs", f"status IN ({_STATUS_OLD})")

        for column in ("next_attempt_at", "max_attempts", "attempts", "tags", "name"):
            if _has_column(bind, "jobs", column):
                op.drop_column("jobs", column)
