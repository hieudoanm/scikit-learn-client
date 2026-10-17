# Local Development

## Prerequisites

- Docker and Docker Compose v2
- Node.js 26+ (frontend host tooling)
- Python 3.14+ (backend host tooling, optional if using Compose only)
- pnpm, uv (optional)
- Git

## Repository structure

```text
skl/
├── packages/
│   ├── backend/      # FastAPI
│   ├── data/         # Sample datasets
│   └── frontend/     # Next.js
├── docker-compose.yaml
├── docs/
└── README.md
```

## Docker Compose services

Defined in [`docker-compose.yaml`](../../docker-compose.yaml). Both images build
from the **repository root** context.

### postgres

- Image: `postgres:18-alpine`
- Port: `127.0.0.1:5432` (host) → `5432` (container), via `POSTGRES_PORT` in `.env`
- Volume: `skl_pg`, mounted at `/var/lib/postgresql` (PostgreSQL 18 keeps its
  cluster in the major-version subdirectory `/var/lib/postgresql/18/docker` and
  refuses to start when it finds a pre-18 cluster at the old
  `/var/lib/postgresql/data` mount point)
- Healthcheck: `pg_isready`
- Env: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` from `.env`

### backend

- Build: `packages/backend/Dockerfile`
- Port: `127.0.0.1:8000`
- Volume: `skl_data:/var/lib/skl`
- Env: `DATABASE_URL`, `skl_DATA_DIR`, `skl_MAX_UPLOAD_MB` (upload cap in MiB, default 10), plus
  `skl_LOG_LEVEL`, `skl_CORS_ORIGINS`, `skl_CORS_ALLOW_CREDENTIALS`, `skl_ENABLE_INTERACTIVE_DOCS`,
  `skl_MAX_CONCURRENT_JOBS`, `skl_JOB_TIMEOUT_SECONDS`, `skl_JOB_MAX_ATTEMPTS`,
  `skl_JOB_RETRY_BASE_SECONDS`, `skl_WORKER_STALE_SECONDS` (see below)
- Depends on: postgres healthy
- Healthcheck: `GET /healthz`

Every variable the backend reads is listed in
[`environment-variables.md`](./environment-variables.md), which is the single source of truth.
Phase 12 centralises configuration in `src/config.py` and wires all `skl_*` variables through
`docker-compose.yaml` so one `.env` controls the whole stack. The CORS allow-list, interactive
docs toggle, logging level, and worker limits are all configurable without code changes.

### frontend

- Build: `packages/frontend/Dockerfile`
- Port: `127.0.0.1:3000`
- Env: `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` (baked in at build time),
  `NEXT_PUBLIC_API_DOCS_URL` (defaults to `${NEXT_PUBLIC_API_URL}/docs` when empty)
- Depends on: backend healthy

## Workflow

```bash
cp .env.example .env
docker compose up --build   # or: make up
```

Open `http://127.0.0.1:3000`. API docs (FastAPI): `http://127.0.0.1:8000/docs`
(Swagger UI) and `http://127.0.0.1:8000/redoc` (ReDoc); the raw schema is at
`http://127.0.0.1:8000/openapi.json`. The committed copy of the schema lives in
[`06-api-specification/backend/openapi.yaml`](../06-api-specification/backend/openapi.yaml).

## CLI

The `skl` CLI is a standalone offline tool (no backend required). Install the
latest release binary:

```bash
curl -fsSL https://github.com/hieudoanm/scikit-learn-client/releases/download/latest/skl \
  -o "$HOME/.local/bin/skl"
chmod +x "$HOME/.local/bin/skl"
```

or run the installer script:

```bash
bash packages/cli/scripts/install.sh
```

Full CLI documentation lives in
[`11-deployment/cli.md`](./cli.md); the CLI source and Makefile are in
[`packages/cli/`](../packages/cli/).

Make shortcuts (from the repository root): `make up` brings up the whole
stack with Docker Compose and `make down` shuts it down. Host mode (optional):
`make start` runs the API and frontend on the host with Postgres via
Compose — the Makefile points `DATABASE_URL` at the Compose-published port
(`POSTGRES_PORT`, 5432 by default), installs the ONNX extra, and passes all `skl_*`
variables from the environment (`.env`); override with `make start POSTGRES_PORT=5433`
if 5432 is already in use. New `skl_*` variables default to the values documented in
[`environment-variables.md`](./environment-variables.md).

## Tests

```bash
make e2e        # Playwright scenarios (backend + frontend must already be up)
make e2e-demo   # same scenarios recorded to demo/demo.mp4 (starts/stops services)
```

## Data

Volumes persist jobs across restarts. Reset:

```bash
# Reset the entire stack including Postgres and artefacts
docker compose down -v && docker compose up --build -d
```

Or use the Makefile:

```bash
make down  # stops and removes containers; add -v to wipe volumes
```

`docker compose down -v` deletes Postgres and artefacts.

PostgreSQL data is not portable across major versions: a volume initialised by
PostgreSQL ≤ 17 cannot be opened by the 18 image (and vice versa). After
bumping the image major version, either reset the volume as above or migrate
the data with `pg_dump` on the old version and `pg_restore` into the new one.
