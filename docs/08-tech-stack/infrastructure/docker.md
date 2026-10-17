# skl — Docker

## Backend

Multi-stage: the builder installs the locked venv with uv (`uv sync --frozen --no-dev --extra onnx`), the runtime stage is `python:3.14.8-slim` running as a non-root user. Entry point: `uvicorn main:app --host 0.0.0.0 --port 8000`. Data lives under `skl_DATA_DIR=/var/lib/skl`.

Build context is the **repository root** so the root `.dockerignore` applies:

```sh
docker build -f packages/backend/Dockerfile -t skl-backend .
```

## Frontend

Multi-stage `node:26.12-alpine` build (matching `.node-version`), runtime `next start --hostname 0.0.0.0 --port 3000`. `NEXT_PUBLIC_API_URL` is inlined at build time, so it is passed as a build arg:

```sh
docker build -f packages/frontend/Dockerfile \
  --build-arg NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 \
  -t skl-frontend .
```

## Compose

Defined in the root [`docker-compose.yaml`](../../../docker-compose.yaml). Services: `postgres` (`postgres:18-alpine`, volume `skl_pg`, healthcheck `pg_isready`), `backend` (builds `packages/backend/Dockerfile`, depends on healthy postgres, volume `skl_data`, healthcheck `GET /healthz`), `frontend` (builds `packages/frontend/Dockerfile`, depends on healthy backend). Ports `127.0.0.1:5432` (Postgres, `POSTGRES_PORT`), `127.0.0.1:8000`, `127.0.0.1:3000`.
