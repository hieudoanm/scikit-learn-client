# Environment Variables

_Part of [11-deployment.md](../11-deployment.md). This document is the **single source of truth**
for every environment variable the project reads._

If a variable is read anywhere in the repository — backend, frontend, or Docker Compose — it is
listed here. If a variable appears in a code path but not in this table, that is a documentation
bug.

## Reading rules

- Every variable has a **default**. A missing variable must never stop the stack from starting.
- Variables are read **once**, at process start, in a single settings module. No module calls
  `os.environ` directly.
- Unknown variables in `.env` are ignored (Compose behaviour) — they are not an error.
- Boolean variables accept `1/0`, `true/false`, `yes/no`, `on/off`, case-insensitively.
- List variables are comma-separated with optional surrounding whitespace; empty items are
  dropped. Example: `skl_CORS_ORIGINS=http://localhost:3000, http://127.0.0.1:3000`.

## Status column

| Status      | Meaning                                                          |
| ----------- | ---------------------------------------------------------------- |
| `Implemented` | Read by the code today.                                        |
| `Planned`     | Specified in the documents, not yet read by the code.          |

---

## Backend

Every backend variable is read exactly once by `packages/backend/src/config.py`, which resolves the
value at process start, falls back to the documented default when the value is unusable, and logs
one warning. The backend container receives these from `docker-compose.yaml`.

| Variable                       | Default                                              | Status       | Purpose                                                                       |
| ------------------------------ | ---------------------------------------------------- | ------------ | ----------------------------------------------------------------------------- |
| `DATABASE_URL`                 | `postgresql://skl:skl@localhost:5432/skl`             | Implemented  | Async SQLAlchemy connection string. The `postgresql+asyncpg://` scheme is normalised to `postgresql://` for asyncpg. In Compose this is built from the `POSTGRES_*` variables. |
| `skl_DATA_DIR`                 | `/var/lib/skl`                                       | Implemented  | Root of the artefact volume: `datasets/{datasetId}.csv` and `artefacts/{jobId}/{jobId}.{ext}`. |
| `skl_MAX_UPLOAD_MB`            | `10`                                                 | Implemented  | Maximum accepted upload size in MiB for `POST /api/v1/datasets`. Over the cap → `413 UPLOAD_TOO_LARGE`. |
| `skl_LOG_LEVEL`                | `INFO`                                               | Implemented  | Root log level for the structured JSON logger (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `skl_CORS_ORIGINS`             | `http://localhost:3000,http://127.0.0.1:3000`        | Implemented  | Comma-separated allow-list of browser origins permitted to call the API. `*` is rejected while credentials are enabled. |
| `skl_CORS_ALLOW_CREDENTIALS`   | `true`                                               | Implemented  | Whether CORS responses may carry `Access-Control-Allow-Credentials`. Must be `false` when `skl_CORS_ORIGINS` contains `*`. |
| `skl_ENABLE_INTERACTIVE_DOCS`  | `true`                                               | Implemented  | Whether the interactive API reference (`/docs`, `/redoc`) and the raw schema (`/openapi.json`) are served. |
| `skl_MAX_CONCURRENT_JOBS`      | `1`                                                  | Implemented  | Maximum number of jobs executing concurrently in the worker. See [`FEATURE-17`](../02-features/FEATURE-17-Reliable-Job-Execution.md). |
| `skl_JOB_TIMEOUT_SECONDS`      | `1800`                                               | Implemented  | Wall-clock budget for a single job attempt. On expiry the job fails with `JOB_TIMEOUT`. |
| `skl_JOB_MAX_ATTEMPTS`         | `2`                                                  | Implemented  | Total attempts per job, including the first. Only transient failures are retried. |
| `skl_JOB_RETRY_BASE_SECONDS`   | `2`                                                  | Implemented  | Base delay of the exponential backoff with full jitter between attempts. |
| `skl_WORKER_STALE_SECONDS`     | `3600`                                               | Implemented  | Age after which a job left in a non-terminal status by a crashed process is marked `failed` with `WORKER_INTERRUPTED`. |

### Notes

- `skl_DATA_DIR` must be an absolute path. A relative path is rejected at startup.
- `skl_CORS_ORIGINS` entries must be scheme + host (+ optional port), with **no** trailing
  slash: `http://127.0.0.1:3000` is valid, `http://127.0.0.1:3000/` is not.
- A CORS origin that does not parse is dropped with a startup warning; if **every** entry is
  dropped, the API starts with an empty allow-list (no cross-origin browser access).
- Values above the documented maxima fall back to the default and log a warning once:
  `skl_MAX_UPLOAD_MB` ≤ 1024, `skl_MAX_CONCURRENT_JOBS` ≤ 16,
  `skl_JOB_TIMEOUT_SECONDS` ≤ 86400, `skl_JOB_MAX_ATTEMPTS` ≤ 5,
  `skl_JOB_RETRY_BASE_SECONDS` ≤ 60, `skl_WORKER_STALE_SECONDS` ≤ 86400.

---

## Frontend

Read by `packages/frontend`. Variables prefixed with `NEXT_PUBLIC_` are inlined into the browser
bundle at build time and are therefore **public**.

| Variable                  | Default                 | Status      | Purpose                                                                 |
| ------------------------- | ----------------------- | ----------- | ----------------------------------------------------------------------- |
| `NEXT_PUBLIC_API_URL`     | `http://127.0.0.1:8000` | Implemented | Base URL of the backend API, used by the browser. Must be reachable **from the browser**, not from inside the container. |
| `NEXT_PUBLIC_API_DOCS_URL`| `${NEXT_PUBLIC_API_URL}/docs` | Implemented | Link to the interactive API reference shown in the app footer. Falls back to `${NEXT_PUBLIC_API_URL}/docs`. |
| `NEXT_PUBLIC_APP_NAME`    | `scikit-learn-client` | Planned | Product name shown in the header and the document title.            |

### Notes

- `NEXT_PUBLIC_API_URL` is a build argument in Compose (`build.args`) as well as a runtime
  environment variable, because the value is inlined at build time. Changing it requires a
  frontend rebuild (`docker compose up --build frontend`).
- `NEXT_PUBLIC_API_DOCS_URL` is inlined the same way; leave it empty to use
  `${NEXT_PUBLIC_API_URL}/docs`.
- Never put a secret in a `NEXT_PUBLIC_` variable.

---

## Docker Compose

Read by `docker-compose.yaml` (and therefore by `.env`). These configure the **PostgreSQL**
service and are interpolated into the backend's `DATABASE_URL`.

| Variable           | Default | Status      | Purpose                                                                     |
| ------------------ | ------- | ----------- | --------------------------------------------------------------------------- |
| `POSTGRES_USER`    | `skl`   | Implemented | Superuser created by the `postgres` image.                                   |
| `POSTGRES_PASSWORD`| `skl`   | Implemented | Password for `POSTGRES_USER`.                                               |
| `POSTGRES_DB`      | `skl`   | Implemented | Database created by the `postgres` image and used by the backend.            |
| `POSTGRES_PORT`    | `5432`  | Implemented | Host port bound to `127.0.0.1` for the PostgreSQL container.                 |

### Notes

- `POSTGRES_PASSWORD` is a placeholder for local development only. It must not be reused outside
  the local stack.
- `POSTGRES_PORT` changes only the **host** port. The container port stays `5432`.
- `docker-compose.yaml` passes every `skl_*` backend variable into the backend container, so a
  single `.env` controls the whole stack.

---

## Test environment

| Variable          | Default                                            | Status      | Purpose                                                    |
| ----------------- | -------------------------------------------------- | ----------- | ---------------------------------------------------------- |
| `DATABASE_URL`    | `postgresql://skl:skl@localhost:5432/skl`          | Implemented | Database used by the backend pytest suite. Identical to CI. |
| `skl_DATA_DIR`    | a per-run temporary directory                      | Implemented | Artefact root for tests; each test uses a fresh directory. |
| `skl_MAX_UPLOAD_MB` | `10`                                             | Implemented | Upload cap asserted by the upload-limit tests.            |

Tests that need a non-default value set it explicitly in the test itself (or through a fixture),
not through the developer's shell.

---

## Where each variable is used

| Variable                      | Read by                                        | Documented in                                 |
| ----------------------------- | ---------------------------------------------- | --------------------------------------------- |
| `DATABASE_URL`                | `src/config.py` → `src/db/session.py`          | [`04-architecture.md`](../04-architecture.md) §4.4 |
| `skl_DATA_DIR`                | `src/config.py` → `src/storage/filesystem.py`  | [`05-data-model.md`](../05-data-model.md) §5   |
| `skl_MAX_UPLOAD_MB`           | `src/config.py` → `src/api/datasets.py`        | [`FEATURE-01`](../02-features/FEATURE-01-Dataset-Upload.md) |
| `skl_LOG_LEVEL`               | `src/config.py` → `main.py`                    | [`FEATURE-17`](../02-features/FEATURE-17-Reliable-Job-Execution.md) |
| `skl_CORS_ORIGINS`            | `src/config.py` → `main.py`                    | [`FEATURE-17`](../02-features/FEATURE-17-Reliable-Job-Execution.md), [`09-testing-strategy.md`](../09-testing-strategy.md) |
| `skl_CORS_ALLOW_CREDENTIALS`  | `src/config.py` → `main.py`                    | [`FEATURE-17`](../02-features/FEATURE-17-Reliable-Job-Execution.md) |
| `skl_ENABLE_INTERACTIVE_DOCS` | `src/config.py` → `main.py`                    | [`FEATURE-18`](../02-features/FEATURE-18-Client-Resilience-and-Guidance.md) |
| `skl_MAX_CONCURRENT_JOBS`     | `src/config.py` → `src/training/worker/worker.py` | [`FEATURE-17`](../02-features/FEATURE-17-Reliable-Job-Execution.md) |
| `skl_JOB_TIMEOUT_SECONDS`     | `src/config.py` → `src/training/worker/worker.py` | [`FEATURE-17`](../02-features/FEATURE-17-Reliable-Job-Execution.md) |
| `skl_JOB_MAX_ATTEMPTS`        | `src/config.py` → `src/training/worker/worker.py` | [`FEATURE-17`](../02-features/FEATURE-17-Reliable-Job-Execution.md) |
| `skl_JOB_RETRY_BASE_SECONDS`  | `src/config.py` → `src/training/worker/worker.py` | [`FEATURE-17`](../02-features/FEATURE-17-Reliable-Job-Execution.md) |
| `skl_WORKER_STALE_SECONDS`    | `src/config.py` → `src/training/worker/worker.py` | [`04-architecture.md`](../04-architecture.md) §8 |
| `NEXT_PUBLIC_API_URL`         | `src/lib/api.ts`                               | [`06-api-specification.md`](../06-api-specification.md) §3 |
| `NEXT_PUBLIC_API_DOCS_URL`    | `src/components/layout/Footer.tsx`             | [`FEATURE-18`](../02-features/FEATURE-18-Client-Resilience-and-Guidance.md) |

---

## Adding a variable

1. Add it to this table with a default, a status, and a purpose.
2. Read it in the backend settings module (or the frontend's config module) — never inline.
3. Add it to `.env.example` if a user is expected to change it.
4. Wire it through `docker-compose.yaml` if the container needs it.
5. Cover the fallback behaviour with a test.

**See also:** [`local-development.md`](./local-development.md),
[`ci-cd-pipeline.md`](./ci-cd-pipeline.md),
[`04-architecture.md`](../04-architecture.md),
[`12-operations.md`](../12-operations.md).
