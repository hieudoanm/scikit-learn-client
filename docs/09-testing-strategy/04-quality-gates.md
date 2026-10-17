# skl — Quality Gates

Pre-commit (Husky): `pnpm turbo run lint` then `pnpm turbo run format` —
whole-workspace, writing in place; `lint-staged` (staged-file checks) is declared
but not wired.

CI is the merge gate:

- `backend` — Ruff lint + format check (`src/` only) and `uv run pytest tests/ -v`
  against a `postgres:18` service.
- `frontend` — ESLint, `tsc --noEmit`, `pnpm run test:coverage`, `next build`.
- `e2e` — Playwright against the running stack (needs `backend` + `frontend`).

The **80%** frontend coverage threshold (statements, branches, functions, lines)
is enforced whenever coverage runs: `pnpm run test:coverage` fails the job below
it, and CI runs that command in `frontend`. The backend carries the same 80% gate
through `pytest-cov` (`--cov-fail-under=80` in
`packages/backend/pyproject.toml`), applied by the `uv run pytest tests/ -v`
step in CI.
