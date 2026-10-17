# AGENTS.md — scikit-learn-client (skl)

Guidance for AI agents (and orientation for humans) working in this repository.

## Read the specification first

The product and implementation specification lives in `docs/`
(`/Users/hieudoan/git/github.com/hieudoanm/skl/docs`). Start at the index,
[`docs/README.md`](docs/README.md). The numbered documents `00`–`12` are
normative: where the index or the code disagrees with them, the numbered
documents govern. When you change behaviour, versions, or ports, update the
affected documents in the same change.

| I need to…                          | Read                                                                                                                                                                   |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Understand vision and scope         | [`docs/00-overview.md`](docs/00-overview.md)                                                                                                                           |
| Check requirements (`FR-`, `NFR-`)  | [`docs/01-requirements.md`](docs/01-requirements.md)                                                                                                                   |
| Understand user-visible behaviour   | [`docs/02-features.md`](docs/02-features.md) (`FEATURE-01`…`18`)                                                                                                       |
| Change architecture/API/data model  | [`docs/04-architecture.md`](docs/04-architecture.md), [`docs/05-data-model.md`](docs/05-data-model.md), [`docs/06-api-specification.md`](docs/06-api-specification.md) |
| Style the UI (tokens, states, a11y) | [`docs/07-ui-ux-design.md`](docs/07-ui-ux-design.md)                                                                                                                   |
| Add or decide on a dependency       | [`docs/08-tech-stack.md`](docs/08-tech-stack.md)                                                                                                                       |
| Know how tests are organised        | [`docs/09-testing-strategy.md`](docs/09-testing-strategy.md)                                                                                                           |
| Touch CI or local setup             | [`docs/11-deployment/ci-cd-pipeline.md`](docs/11-deployment/ci-cd-pipeline.md), [`docs/11-deployment/local-development.md`](docs/11-deployment/local-development.md)   |
| Operate: backup, rollback, security | [`docs/12-operations/`](docs/12-operations/)                                                                                                                           |

## Repository layout

| Path                 | What it is                                                                                                                  |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `packages/backend`   | FastAPI training API (Python, uv, SQLAlchemy async + asyncpg, scikit-learn, ONNX export). See `packages/backend/AGENTS.md`. |
| `packages/frontend`  | Next.js App Router UI (TypeScript, Tailwind CSS, DaisyUI, Playwright). See `packages/frontend/AGENTS.md`.                   |
| `packages/notebooks` | Reproducible Jupyter notebooks (`pipeline.ipynb`, `datasets.ipynb`) and the sample datasets they export.                    |
| `docs/`              | The specification — source of truth (see above).                                                                            |
| `demo/`              | Recorded demo video, produced by `make e2e-demo`.                                                                           |
| `.github/workflows/` | CI: backend (ruff + pytest), frontend (eslint + tsc + build), e2e (Playwright) — all against a `postgres:18` service.       |

pnpm workspace (`pnpm-workspace.yaml`) driven by turbo: `pnpm run dev | lint | build | format | typecheck | test` (`format` and `typecheck` run Prettier/`tsc` for the frontend and Ruff for the Python packages; `test` runs the Vitest suite in `packages/frontend`).

## Common commands

```sh
# Full local stack: Postgres 18 on 127.0.0.1:5432, backend :8000, frontend :3000
docker compose up -d --build

# The same stack via make
make up        # / make down

# Host mode instead: Postgres via compose, backend/frontend on the host
make start   # / make stop

# Backend API e2e (needs the stack up)
cd packages/backend && uv run pytest tests/ -v

# Playwright e2e (needs backend :8000 + frontend :3000)
make e2e
# or: cd packages/frontend && pnpm exec playwright test

# Frontend unit tests (Vitest + React Testing Library; `test:coverage` is the
# 80% gate that CI runs)
cd packages/frontend && pnpm test
```

## Environment and conventions

- **Node 26** (`.node-version`, `.nvmrc`) with pnpm. **Python 3.14.8**
  (`packages/backend/.python-version` pins **3.14.8**) — CI, the Docker image,
  and `make start` all use it.
- **PostgreSQL 18** everywhere (compose, CI, docs). Compose mounts the volume
  at `skl_pg:/var/lib/postgresql` — PG 18 images keep data in an `18/docker`
  subdirectory and refuse to start on a pre-18 data directory at the old
  `/var/lib/postgresql/data` mount, so a major-version upgrade means
  dump → fresh volume → restore (see
  [`docs/11-deployment/local-development.md`](docs/11-deployment/local-development.md)).
- `.env` (copied from `.env.example`, gitignored) parameterises compose:
  `POSTGRES_PORT` (default **5432**), `POSTGRES_USER/PASSWORD/DB`.
- Backend tests read `DATABASE_URL` (default
  `postgresql://skl:skl@localhost:5432/skl`, identical to CI); the
  `postgresql+asyncpg://` scheme is normalised for asyncpg.
- ONNX export needs the optional `onnx` extra: `uv sync --extra onnx`
  (CI, the Docker image, and `make start` already do this).
- Ruff lives in `packages/backend/pyproject.toml` (line length 100, single
  quotes); CI gates `ruff check` and `ruff format --check` on `src/` only.
- Frontend e2e selectors: use role/label locators but avoid ambiguous
  `aria-label` values — Playwright strict mode fails on multiple matches.
- Keep ports and versions you touch in sync across `docker-compose.yaml`,
  `.env.example`, `Makefile`, `docs/`, and `README.md`.
