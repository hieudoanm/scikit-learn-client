"""Phase 9: nullable label column and the five-value task-type check.

Revision ID: 0001
Revises:
Create Date: 2026-10-19

This is the first migration. ``init_db()`` creates the schema from the
SQLAlchemy metadata at startup, so on a fresh database this revision is a
no-op (the tables do not exist yet and the metadata already carries the new
definition). On an existing local volume it upgrades ``jobs`` in place.

Downgrade restores the pre-Phase-9 definition and therefore fails if the table
still contains semi-supervised or unsupervised jobs; delete those rows first.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None

TASK_TYPES = (
    'classification',
    'regression',
    'semi_supervised',
    'clustering',
    'anomaly_detection',
)
LABEL_REQUIRED = ('classification', 'regression', 'semi_supervised')
NO_LABEL = ('clustering', 'anomaly_detection')


def _in_list(values: tuple[str, ...]) -> str:
    return ', '.join(f"'{value}'" for value in values)


TASK_CHECK = f'task_type IN ({_in_list(TASK_TYPES)})'
LABEL_CHECK = (
    f'(task_type IN ({_in_list(LABEL_REQUIRED)}) AND label_column IS NOT NULL) OR '
    f'(task_type IN ({_in_list(NO_LABEL)}) AND label_column IS NULL)'
)


def _has_jobs(bind) -> bool:
    return 'jobs' in sa.inspect(bind).get_table_names()


def _check_names(bind) -> set[str]:
    return {constraint['name'] for constraint in sa.inspect(bind).get_check_constraints('jobs')}


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_jobs(bind):
        return

    checks = _check_names(bind)
    op.alter_column(
        'jobs',
        'label_column',
        existing_type=sa.String(length=255),
        nullable=True,
    )
    if 'jobs_task_type_check' in checks:
        op.drop_constraint('jobs_task_type_check', 'jobs', type_='check')
    op.create_check_constraint('jobs_task_type_check', 'jobs', TASK_CHECK)
    if 'jobs_label_column_matches_task' not in checks:
        op.create_check_constraint('jobs_label_column_matches_task', 'jobs', LABEL_CHECK)


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_jobs(bind):
        return

    checks = _check_names(bind)
    if 'jobs_label_column_matches_task' in checks:
        op.drop_constraint('jobs_label_column_matches_task', 'jobs', type_='check')
    if 'jobs_task_type_check' in checks:
        op.drop_constraint('jobs_task_type_check', 'jobs', type_='check')
    op.create_check_constraint(
        'jobs_task_type_check',
        'jobs',
        "task_type IN ('classification', 'regression')",
    )
    op.alter_column(
        'jobs',
        'label_column',
        existing_type=sa.String(length=255),
        nullable=False,
    )
