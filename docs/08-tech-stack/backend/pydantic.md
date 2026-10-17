# skl — Pydantic v2

## Usage in skl

Pydantic v2 is the **validation and serialisation layer** for the entire API:

- All request/response bodies are typed Pydantic models
- FastAPI derives the OpenAPI contract from these models
- Models live in `src/api/schemas.py` (re-exported from `src/api/__init__.py`)
- Every endpoint signature uses `response_model=` and typed parameters

## Model catalogue

| Model | Purpose | Endpoint |
|-------|---------|----------|
| `HealthResponse`, `ReadyResponse` | Liveness/readiness probes | `GET /healthz`, `GET /readyz` |
| `DatasetResponse`, `ColumnPreview` | CSV upload result + metadata | `POST/GET /api/v1/datasets` |
| `AlgorithmResponse`, `AlgorithmsListResponse` | Algorithm catalogue | `GET /api/v1/algorithms` |
| `CreateJobRequest` | Enqueue training (validated) | `POST /api/v1/jobs` |
| `JobAcceptedResponse` | 202 response | `POST /api/v1/jobs`, `GET /api/v1/jobs` (list) |
| `JobResponse` | Full job detail | `GET /api/v1/jobs/{jobId}` |
| `JobListResponse` | Paginated history | `GET /api/v1/jobs` |
| `ArtefactResponse` | Download metadata | `GET /api/v1/jobs/{jobId}/artefacts/{artefactId}` |
| `RLColumns` | RL experience columns | Nested in `CreateJobRequest` |
| `ErrorDetail`, `ErrorEnvelope` | Unified error shape | All error responses |

## Enums (mirror OpenAPI)

```python
class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEMI_SUPERVISED = "semi_supervised"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

class JobStatus(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    TRAINING = "training"
    EVALUATING = "evaluating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"

class ArtefactFormat(StrEnum):
    ONNX = "onnx"
    PKL = "pkl"
    JOBLIB = "joblib"
```

## Cross-field validation

`CreateJobRequest` uses `@model_validator(mode="after")` to enforce label/rlColumns rules per task type:

```python
@model_validator(mode="after")
def validate_label_rules(self) -> "CreateJobRequest":
    label_required = {TaskType.CLASSIFICATION, TaskType.REGRESSION, TaskType.SEMI_SUPERVISED}
    label_forbidden = {TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION, TaskType.REINFORCEMENT_LEARNING}

    if self.taskType in label_required:
        if not self.labelColumn:
            raise ValueError(f"labelColumn required for {self.taskType.value}")
    if self.taskType in label_forbidden:
        if self.labelColumn:
            raise ValueError(f"{self.taskType.value} must not set labelColumn")
    # ... RL rules
    return self
```

This keeps validation **at the boundary** — invalid payloads are rejected before reaching business logic.

## Integration with `validate_job_request()`

The helper `src/api/jobs_validation.validate_job_request()` accepts both `CreateJobRequest` and `dict`:

```python
async def validate_job_request(
    session,
    body: CreateJobRequest | dict,
) -> tuple[UUID, Dataset, dict]:
    if isinstance(body, CreateJobRequest):
        body_dict = body.model_dump()
    else:
        body_dict = body
    # ... database validation, hyperparameter checks, etc.
```

This preserves backward compatibility with existing tests that pass raw dicts.

## Configuration

```python
class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # ...
```

- `extra="forbid"` on `CreateJobRequest` and the nested `RLColumns` — unknown keys (for example the old `exportFormats` array) are rejected with `400 VALIDATION_ERROR`
- `Field(ge=0, lt=1)` for numeric constraints (`splitRatio`, `randomSeed`)
- `UUID` and `datetime` types auto-coerce from strings

## Error envelope

Request-validation failures are normalised into the same flat envelope as application errors by the handlers in `main.py` (see [`docs/06`](../../06-api-specification.md#6-error-response-format)):

```json
{ "error": { "code": "VALIDATION_ERROR", "message": "...", "requestId": "uuid", "details": { "fields": [] } } }
```

## Testing

- Unit tests construct `CreateJobRequest` directly to verify validation
- Error messages are asserted via `pytest.raises(ValidationError)`
- E2E tests hit the real FastAPI endpoints (OpenAPI contract verified)

## Version pin

`pydantic>=2.10,<3` in `pyproject.toml` (via `uv.lock`).

## Rejected alternatives

| Alternative | Reason |
|-------------|--------|
| v1 `BaseModel` | No `model_validator`, slower, no `ConfigDict` |
| marshmallow | No FastAPI integration, separate OpenAPI step |
| attrs + cattrs | Verbose, no built-in OpenAPI |
| Raw dict + manual validation | Error-prone, no contract |