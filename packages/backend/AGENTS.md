# AGENTS.md — backend (`packages/backend`)

FastAPI service that accepts a CSV upload, trains a scikit-learn model, and
exports ONNX. The specification in `docs/`
(`/Users/hieudoan/git/github.com/hieudoanm/skl/docs`) governs.

## Specification first

- [`docs/06-api-specification.md`](../../docs/06-api-specification.md) — REST
  contract, status codes, error envelope. Don't change routes or response
  shapes without updating it (and the frontend client).
- [`docs/05-data-model.md`](../../docs/05-data-model.md) and
  [`docs/05-data-model/postgresql/schema.sql`](../../docs/05-data-model/postgresql/schema.sql)
  — schema, constraints, job state transitions.
- [`docs/04-architecture.md`](../../docs/04-architecture.md) — module
  boundaries and the request/training lifecycle.
- [`docs/09-testing-strategy.md`](../../docs/09-testing-strategy.md) — test
  layers and fixtures;
  [`docs/11-deployment/local-development.md`](../../docs/11-deployment/local-development.md)
  — how to run the stack.

The catalogue spans five task types (`classification`, `regression`,
`semi_supervised`, `clustering`, `anomaly_detection`). `jobs.label_column` is
nullable and constrained by task type; semi-supervised label cells may be empty
(`-1` sentinel internally); IsolationForest export pins `ai.onnx.ml` to 3. See
`docs/02-features/FEATURE-09-Semi-Supervised-Learning.md` and
`docs/02-features/FEATURE-10-Unsupervised-Learning.md`.

## Layout

- `main.py` — service entry (`uvicorn main:app`), run from this directory.
  `create_app()` builds the application; the CORS allow-list and the
  interactive-documentation switch come from `src/config.py`.
- `src/config.py` — the single settings module. Every environment variable is
  read here, once, at process start; an unusable value falls back to the
  documented default with one warning. See
  `docs/11-deployment/environment-variables.md`.
- `src/api/` — routers (`datasets.py`, `jobs.py`, `jobs_lifecycle.py`,
  `algorithms.py`, `health.py`) plus their helpers. `jobs.py` owns the discovery
  views (history, comparison, multi-job bundle); `jobs_lifecycle.py` owns create,
  poll, rename, delete, and artefact download. Schemas live in `src/api/schemas/`
  grouped by domain and re-exported from its `__init__.py`.
- `src/db/` — engine/session and models. `init_db()` creates the schema at
  startup via `Base.metadata.create_all`.
- `src/storage/`, `src/training/` (worker), `src/export/`
  (ONNX and optional formats).
- `src/insights.py` — the per-dataset column profile.
- `src/api/errors.py` — the unhandled-exception classifier that keeps driver
  text out of responses (`DATABASE_ERROR`/`CONFLICT`).
- `scripts/export_openapi.py` — regenerates
  `docs/06-api-specification/backend/openapi.yaml` from the app; `--check`
  fails when the committed document has drifted (CI runs it).
- `tests/` — grouped by module (`api/`, `db/`, `export/`, `storage/`,
  `training/`) plus the API-level e2e `test_e2e.py` (needs a running server);
  `conftest.py` resolves `DATABASE_URL`.

## Run it

```sh
# Database (from the repository root)
docker compose up -d postgres      # PostgreSQL 18 on 127.0.0.1:5432

# API (from this directory)
DATABASE_URL=postgresql+asyncpg://skl:skl@localhost:5432/skl \
SKL_DATA_DIR=/tmp/skl-data \
uv run --extra onnx python -m uvicorn main:app --host 127.0.0.1 --port 8000

# Or the whole stack from the repository root (Docker Compose)
make up
# Or backend + frontend on the host instead (Postgres via Compose)
make start
```

Schema is created at startup by `init_db()`. Alembic is wired up
(`alembic.ini`, `migrations/`) with revision `0005` in
`migrations/versions/` (the `retrying` status, the discovery columns, and
`dataset_insights`); on a fresh database run by `init_db()` it is a no-op, and
on an existing volume `alembic upgrade head` (with `DATABASE_URL` set) brings
the schema forward.
`alembic.ini` sets `prepend_sys_path = .`, so run Alembic from this directory.
When you add migrations, follow `docs/05-data-model.md` and remember the
volume-layout note in `docs/11-deployment/local-development.md` for upgrades
across PostgreSQL major versions.

## Tests

```sh
uv run pytest tests/ -v
```

- Requires Postgres reachable at `DATABASE_URL` (compose publishes it on
  **5432**, the same port CI uses). `tests/test_e2e.py` additionally needs the
  API on `127.0.0.1:8000` — CI starts `uvicorn` itself, so locally run
  `docker compose up -d` (or start the API as above) first. The other
  directories use the in-process `TestClient`.
- `conftest.py::_database_url()` reads `DATABASE_URL` (default
  `postgresql://skl:skl@localhost:5432/skl`, identical to CI) and normalises
  `postgresql+asyncpg://` → `postgresql://` for asyncpg.
- Coverage is enforced: `addopts` in `pyproject.toml` runs `pytest-cov` over
  `src/` with `--cov-fail-under=80` and writes `htmlcov/`. CI's `backend` job
  fails below 80%.
- Covered flows: upload → create job → poll → artefact download → delete
  (see `docs/09-testing-strategy/05-critical-test-scenarios.md`).

## Python, uv, lint

- `.python-version` pins **3.14.8**, `requires-python` is `">=3.14"`, and CI,
  the Docker image, and `make start` all run Python 3.14.
- ONNX support is the optional `onnx` extra in `pyproject.toml`: always
  `uv sync --extra onnx` (or `uv run --extra onnx`), otherwise export fails.
- Ruff config is in `pyproject.toml` (line length 100, single quotes).
  CI gates exactly:

  ```sh
  uv run ruff check src/
  uv run ruff format --check src/
  ```

  Only `src/` is lint-gated; keep `tests/` consistent in style anyway.

- Dependencies are pinned exactly in `pyproject.toml` and `uv.lock`;
  dependency policy in [`docs/08-tech-stack.md`](../../docs/08-tech-stack.md).
