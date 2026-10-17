# skl — Backend

FastAPI service that validates CSV uploads, persists job metadata, trains
scikit-learn models asynchronously, and exports **ONNX** artefacts (plus optional
pickle and joblib).

## Stack

Python 3.14 · FastAPI · Pydantic · SQLAlchemy (async) · Alembic · scikit-learn ·
ONNX (`skl2onnx`) · PostgreSQL 18.

## Layout

```text
main.py            # FastAPI app + `uvicorn main:app` entry point
src/
├── api/           # health, datasets, jobs, algorithms routers
├── db/            # async engine, session, models, utils
├── export/        # ONNX / pickle / joblib exporters
├── schemas/       # Pydantic request/response models
├── storage/       # filesystem dataset and artefact storage
└── training/      # training tasks and background worker
migrations/        # Alembic
tests/             # pytest
```

## Configuration

| Variable       | Default                                           | Purpose                                                                                          |
| -------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `DATABASE_URL` | `postgresql+asyncpg://skl:skl@localhost:5432/skl` | Async SQLAlchemy URL. `postgres://` / `postgresql://` are normalised to `postgresql+asyncpg://`. |
| `SKL_DATA_DIR` | `/var/lib/skl`                                    | Root directory for `datasets/` and `artefacts/`.                                                 |

## Run locally

Requires Python 3.14+, [uv](https://docs.astral.sh/uv/), and PostgreSQL 18.

```sh
# 1. Install dependencies (including the ONNX extra)
uv sync --extra onnx

# 2. Start PostgreSQL (from the repository root; published on 5432 by default)
docker compose up -d postgres

# 3. Run the API
DATABASE_URL=postgresql+asyncpg://skl:skl@localhost:5432/skl \
SKL_DATA_DIR=/tmp/skl-data \
uv run python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

> Compose maps the container's PostgreSQL to host port `5432` (`POSTGRES_PORT`
> in `.env`). Set `POSTGRES_PORT=5432` to keep 5432 free for another local
> PostgreSQL instead.

From the repository root, `make up` brings up the full stack with Docker
Compose (database, API, frontend); `make start` runs the API and frontend
on the host instead.

## Endpoints

| Method   | Path                                            | Purpose                                    |
| -------- | ----------------------------------------------- | ------------------------------------------ |
| `GET`    | `/healthz`                                      | Liveness probe.                            |
| `GET`    | `/readyz`                                       | Readiness probe (checks PostgreSQL).       |
| `GET`    | `/api/v1/algorithms`                            | Algorithm catalogue (all task types).      |
| `POST`   | `/api/v1/datasets`                              | Upload a CSV; returns columns and preview. |
| `GET`    | `/api/v1/datasets/{dataset_id}`                 | Dataset metadata.                          |
| `POST`   | `/api/v1/jobs`                                  | Submit a training job.                     |
| `GET`    | `/api/v1/jobs`                                  | List training jobs.                        |
| `GET`    | `/api/v1/jobs/{job_id}`                         | Poll job status, metrics, artefacts.       |
| `DELETE` | `/api/v1/jobs/{job_id}`                         | Delete a job and its files.                |
| `GET`    | `/api/v1/jobs/{job_id}/artefacts/{artefact_id}` | Download an artefact.                      |

Interactive docs: <http://127.0.0.1:8000/docs>.

## Quality gates

```sh
uv run ruff check src/
uv run ruff format --check src/
uv run pytest tests/ -v
```
