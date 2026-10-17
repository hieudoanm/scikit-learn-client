# skl — Backend Best Practices

## Documents

| Document                               | Covers                           |
| -------------------------------------- | -------------------------------- |
| [`python.md`](./python.md)             | Layout, typing, Ruff             |
| [`fastapi.md`](./fastapi.md)           | Routers, errors, uploads         |
| [`pydantic.md`](./pydantic.md)         | Validation, serialisation, OpenAPI |
| [`scikit-learn.md`](./scikit-learn.md) | Pipelines, metrics               |
| [`models.md`](./models.md)             | The 25-estimator model catalogue |
| [`neural-networks.md`](./neural-networks.md) | Tabular MLPs (P1)         |
| [`onnx.md`](./onnx.md)                 | Required export                  |
| [`postgresql.md`](./postgresql.md)     | SQLAlchemy, worker claim         |

## Shared rules

1. Fit/export off the event loop (`asyncio.to_thread`).
2. Allow-listed `algorithmId` only.
3. Never `pickle.loads` user bytes.
4. Parameterised SQL.
5. Completed jobs have ONNX or they are not completed.
6. Structured logs carry `requestId` (from the request-scoped context) and `jobId` where relevant.
7. Relative storage keys.
8. Pydantic models match OpenAPI.
