# skl — Python

Package: `packages/backend`. Layout:

```
main.py            # FastAPI app + `uvicorn main:app` entry point
src/
  api/             # health, datasets, jobs, algorithms routers
  db/              # async engine, session, models, utils
  export/          # ONNX / pickle / joblib exporters
  schemas/         # Pydantic request/response models
  storage/         # filesystem dataset and artefact storage
  training/        # training tasks and background worker
migrations/        # Alembic
tests/
```

Python 3.14, Ruff (`lint` + `format`), `mypy` or `ty` optional but `pydantic` models must be typed. Lock with uv. No notebooks in the service image.

Lint/format rules: [`tools/ruff.md`](../tools/ruff.md).
