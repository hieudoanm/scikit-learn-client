# skl — PostgreSQL access

SQLAlchemy 2.x. Alembic migrations equivalent to [`schema.sql`](../../05-data-model/postgresql/schema.sql). Connection string from env `DATABASE_URL`. Pool size small (5). Worker claim: `FOR UPDATE SKIP LOCKED`. JSONB loads into Pydantic. Tests use a real Postgres service container, not SQLite.
