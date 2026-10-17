# skl — GitHub Actions

Workflow: [`.github/workflows/ci.yaml`](../../../.github/workflows/ci.yaml) on `push` and `pull_request` to `master`.

1. `backend`: `uv sync --extra onnx`, Ruff (`check src/` + `format --check src/`), then pytest against a `postgres:18` service container (server started with `uvicorn`) — 80% coverage gate via `pytest-cov`.
2. `frontend`: `pnpm install --frozen-lockfile`, `pnpm run lint`, `pnpm exec tsc --noEmit`, `pnpm run test:coverage` (80% gate), `pnpm run build`.
3. `e2e`: needs both jobs; starts the API and `pnpm run dev` with a `postgres:18` service, runs Playwright, uploads `playwright-report/`.

Pinned toolchain: Python 3.14, Node 26, pnpm 12.14.0 — `setup-uv` and the pnpm/node caches are enabled. No cloud deploy job. A Trivy filesystem scan is welcome but not a substitute for tests.
