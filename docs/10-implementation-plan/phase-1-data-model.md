# Phase 1: Data model

Week 1. Alembic to [`schema.sql`](../05-data-model/postgresql/schema.sql). `/readyz` pings Postgres. Volume mount `skl_data`.

## Acceptance

Migrations apply on backend start. pytest can insert a dataset row.
