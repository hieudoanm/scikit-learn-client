# skl — FastAPI

- Routers under `/api/v1`.
- Exception handlers emit the error envelope; include `requestId` middleware.
- `UploadFile` spooled to disk; `max` size via Starlette limit matching NFR-030.
- CORS: frontend origin only.
- `GET /healthz` no DB; `GET /readyz` `SELECT 1`.
- Background: startup creates the worker loop task.
