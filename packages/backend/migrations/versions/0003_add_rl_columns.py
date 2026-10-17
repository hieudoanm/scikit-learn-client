"""Add rl_columns to jobs and extend task_type constraints for RL.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-20

Adds the nullable `rl_columns` JSONB column that stores the experience-column
mapping for `reinforcement_learning` jobs (FEATURE-11), and widens the
`jobs_task_type_check` and `jobs_label_column_matches_task` constraints to
accept the new task type. `rl_columns` is NULL for every other task, enforced
by `jobs_rl_columns_matches_task`.

Downgrade removes the column and restores the pre-RL constraints; RL job rows
cannot survive a downgrade, so they are deleted first.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_TASK_TYPES_OLD = (
    "'classification', 'regression', 'semi_supervised', "
    "'clustering', 'anomaly_detection'"
)
_TASK_TYPES_NEW = _TASK_TYPES_OLD + ", 'reinforcement_learning'"

_LABEL_OLD = (
    "(task_type IN ('classification', 'regression', 'semi_supervised') "
    "AND label_column IS NOT NULL) OR "
    "(task_type IN ('clustering', 'anomaly_detection') AND label_column IS NULL)"
)
_LABEL_NEW = (
    "(task_type IN ('classification', 'regression', 'semi_supervised') "
    "AND label_column IS NOT NULL) OR "
    "(task_type IN ('clustering', 'anomaly_detection', 'reinforcement_learning') "
    "AND label_column IS NULL)"
)
_RL_CHECK = (
    "(task_type = 'reinforcement_learning' AND rl_columns IS NOT NULL) OR "
    "(task_type <> 'reinforcement_learning' AND rl_columns IS NULL)"
)


def _has_table(bind, table_name: str) -> bool:
    return table_name in sa.inspect(bind).get_table_names()


def _has_column(bind, table_name: str, column_name: str) -> bool:
    columns = sa.inspect(bind).get_columns(table_name)
    return any(col["name"] == column_name for col in columns)


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "jobs"):
        return

    if not _has_column(bind, "jobs", "rl_columns"):
        op.add_column("jobs", sa.Column("rl_columns", sa.dialects.postgresql.JSONB(), nullable=True))

    op.drop_constraint("jobs_task_type_check", "jobs", type_="check")
    op.create_check_constraint(
        "jobs_task_type_check", "jobs", f"task_type IN ({_TASK_TYPES_NEW})"
    )

    op.drop_constraint("jobs_label_column_matches_task", "jobs", type_="check")
    op.create_check_constraint("jobs_label_column_matches_task", "jobs", _LABEL_NEW)

    op.create_check_constraint("jobs_rl_columns_matches_task", "jobs", _RL_CHECK)


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "jobs"):
        return

    op.execute("DELETE FROM jobs WHERE task_type = 'reinforcement_learning'")

    op.drop_constraint("jobs_rl_columns_matches_task", "jobs", type_="check")

    op.drop_constraint("jobs_label_column_matches_task", "jobs", type_="check")
    op.create_check_constraint("jobs_label_column_matches_task", "jobs", _LABEL_OLD)

    op.drop_constraint("jobs_task_type_check", "jobs", type_="check")
    op.create_check_constraint(
        "jobs_task_type_check", "jobs", f"task_type IN ({_TASK_TYPES_OLD})"
    )

    if _has_column(bind, "jobs", "rl_columns"):
        op.drop_column("jobs", "rl_columns")
