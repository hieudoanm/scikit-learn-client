"""Remove the phantom `export_keras` column and the `keras` artefact format.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-20

`keras` was declared in the schema (`jobs.export_keras`,
`artefacts_format_check`) and the request model (`exportKeras`) but no worker or
export path ever produced a Keras artefact, so the API advertised a format it
could not deliver. This revision drops the column and narrows the CHECK to the
formats the pipeline actually writes (`onnx`, `pkl`, `joblib`), matching
docs/05-data-model/postgresql/schema.sql.

`init_db()` builds a fresh database from the metadata, so this is a no-op there
(the new metadata already omits the column); it upgrades existing volumes in
place. Downgrade restores the pre-0004 definition.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

_FORMATS_OLD = "'onnx', 'pkl', 'joblib', 'keras'"
_FORMATS_NEW = "'onnx', 'pkl', 'joblib'"


def _has_table(bind, table_name: str) -> bool:
    return table_name in sa.inspect(bind).get_table_names()


def _has_column(bind, table_name: str, column_name: str) -> bool:
    columns = sa.inspect(bind).get_columns(table_name)
    return any(col["name"] == column_name for col in columns)


def upgrade() -> None:
    bind = op.get_bind()

    if _has_table(bind, "jobs") and _has_column(bind, "jobs", "export_keras"):
        op.drop_column("jobs", "export_keras")

    if _has_table(bind, "artefacts"):
        op.drop_constraint("artefacts_format_check", "artefacts", type_="check")
        op.create_check_constraint(
            "artefacts_format_check", "artefacts", f"format IN ({_FORMATS_NEW})"
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _has_table(bind, "jobs") and not _has_column(bind, "jobs", "export_keras"):
        op.add_column(
            "jobs",
            sa.Column(
                "export_keras",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )

    if _has_table(bind, "artefacts"):
        op.drop_constraint("artefacts_format_check", "artefacts", type_="check")
        op.create_check_constraint(
            "artefacts_format_check", "artefacts", f"format IN ({_FORMATS_OLD})"
        )
