# CI/CD Pipeline

GitHub Actions only ([`.github/workflows/ci.yaml`](../../.github/workflows/ci.yaml)). No deploy job.

Triggers: `push` and `pull_request` on `master`. Toolchain: Python 3.14, Node 26, pnpm 12.14.0.

| Job        | What it runs                                                                                                                                                                                                                           |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend`  | `uv sync --extra onnx`, `uv run ruff check src/`, `uv run ruff format --check src/`, `uv run python scripts/export_openapi.py --check` (OpenAPI drift check), then starts `uvicorn` against a `postgres:18` service container and runs `uv run pytest tests/ -v` (80% coverage gate via `pytest-cov` `addopts`) |
| `frontend` | `pnpm install --frozen-lockfile`, `pnpm run lint`, `pnpm exec tsc --noEmit`, `pnpm run test:coverage` (80% gate), `pnpm run build`                                                                                                     |
| `e2e`      | needs `backend` + `frontend`; starts the API and `pnpm run dev` against a `postgres:18` service, installs Playwright browsers, runs `pnpm exec playwright test`, uploads `playwright-report/`                                          |

Both test jobs fail below **80%** coverage — Vitest for the frontend, `pytest-cov`
for the backend. See
[`09-testing-strategy/04-quality-gates.md`](../09-testing-strategy/04-quality-gates.md).

The `backend` job also fails when the committed OpenAPI document drifts away from the code.
`packages/backend/scripts/export_openapi.py` regenerates
`docs/06-api-specification/backend/openapi.yaml` from the FastAPI application; `--check` compares
without writing. After changing a route or a schema, run:

```sh
cd packages/backend && uv run python scripts/export_openapi.py
cd ../../docs/06-api-specification && uv run openapi.py   # refresh the JSON + Postman files
```

OIDC/cloud credentials are not used.
