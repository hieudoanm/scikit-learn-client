# Rollback Strategy

Images are built locally. Rollback = git checkout previous revision and `docker compose up --build`.

Alembic: prefer backward-compatible migrations; if a down migration exists, `alembic downgrade -1` before starting old code. Never leave `completed` jobs without files after a volume wipe.

Frontend/backend are independent images; a frontend rollback against a newer API must still speak documented OpenAPI — pin API versions (`/api/v1`).
