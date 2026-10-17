"""skl backend FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from src.api import algorithms, datasets, health, jobs, jobs_lifecycle
from src.api.deps import build_error_body
from src.api.errors import classify_unhandled
from src.api.request_context import get_request_id
from src.config import Settings, get_settings
from src.db.session import init_db
from src.logging.config import get_logger, setup_logging
from src.middleware import setup_middleware
from src.training.worker import enqueue_job, recover_interrupted_jobs

logger = get_logger('skl.api')

settings = get_settings()

OPENAPI_TAGS = [
    {
        'name': 'health',
        'description': 'Liveness and readiness probes.',
    },
    {
        'name': 'datasets',
        'description': 'CSV upload, dataset metadata, and per-column insights.',
    },
    {
        'name': 'algorithms',
        'description': 'Algorithm catalogue with task types and hyperparameter ranges.',
    },
    {
        'name': 'jobs',
        'description': 'Training jobs: creation, history, comparison, and artefact download.',
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB and recover interrupted jobs on startup."""
    setup_logging(settings.log_level)
    await init_db()
    try:
        requeued = await recover_interrupted_jobs()
    except Exception:  # noqa: BLE001
        logger.exception('Failed to recover the jobs interrupted by the last shutdown')
        requeued = []
    for job_id in requeued:
        enqueue_job(job_id)
    yield


def _validation_details(exc: RequestValidationError) -> dict:
    """Build the ``details`` of a request-validation failure.

    The per-field errors are always present under ``fields``. A field that
    exceeded a documented limit also gets ``details.field`` and
    ``details.limit`` so a client can point at the offending input without
    parsing the Pydantic error list (FEATURE-13).
    """
    errors = exc.errors()
    details: dict = {'fields': errors}
    for error in errors:
        if error.get('type') != 'too_long':
            continue
        location = error.get('loc') or ()
        if location:
            details.setdefault('field', str(location[-1]))
        limit = (error.get('ctx') or {}).get('max_length')
        if limit is not None:
            details.setdefault('limit', limit)
    return details


def create_app(app_settings: Settings | None = None) -> FastAPI:
    """Build the skl FastAPI application.

    The settings are read from the central configuration module; a caller may
    pass an explicit set of settings to build an app with a different CORS
    allow-list or without the interactive documentation.
    """
    resolved = app_settings or get_settings()

    application = FastAPI(
        title='skl Training API',
        version='0.1.0',
        description=(
            'Local FastAPI contract for the scikit-learn-client platform. '
            'ONNX is the required training artefact.'
        ),
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        servers=[{'url': 'http://127.0.0.1:8000', 'description': 'Docker Compose backend'}],
        # The interactive reference is optional so a deployment can expose the API
        # without shipping a browser-friendly schema UI.
        docs_url='/docs' if resolved.enable_interactive_docs else None,
        redoc_url='/redoc' if resolved.enable_interactive_docs else None,
        openapi_url='/openapi.json' if resolved.enable_interactive_docs else None,
    )

    application.include_router(health.router)
    application.include_router(algorithms.router)
    application.include_router(datasets.router)
    # The static job paths (``/jobs/compare``, ``/jobs/artefacts.zip``) are
    # registered before the UUID path parameter so they are never captured.
    application.include_router(jobs.router)
    application.include_router(jobs_lifecycle.router)

    setup_middleware(application, resolved)

    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Emit the skl error envelope at the top level for HTTP exceptions.

        ``raise_http`` attaches the envelope to ``detail``; here it is unwrapped so
        the body is ``{"error": {...}}`` rather than ``{"detail": {"error": ...}}``.
        Any other ``HTTPException`` (e.g. raised by FastAPI itself) is normalised
        into the same shape.
        """
        detail = exc.detail
        if isinstance(detail, dict) and 'error' in detail:
            body = detail
        else:
            body = build_error_body(
                code=f'HTTP_{exc.status_code}',
                message=str(detail),
                request_id=get_request_id(),
            )
        logger.warning(
            '%s %s -> %s %s',
            request.method,
            request.url.path,
            exc.status_code,
            body['error']['code'],
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder(body),
            headers=getattr(exc, 'headers', None),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return FastAPI request-validation failures as the skl error envelope."""
        logger.warning('%s %s -> 400 VALIDATION_ERROR', request.method, request.url.path)
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder(
                build_error_body(
                    code='VALIDATION_ERROR',
                    message='Request validation failed',
                    request_id=get_request_id(),
                    details=_validation_details(exc),
                )
            ),
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return any unhandled error as the skl envelope with `INTERNAL_ERROR` (500).

        Keeps stack traces out of JSON (`FEATURE-07`). Database errors are
        normalised to a generic message so the raw driver text (DSN, SQL,
        constraint names) never reaches the client.
        """
        code, message = classify_unhandled(exc)
        if code == 'DATABASE_ERROR':
            logger.exception('Database error on %s %s', request.method, request.url.path)
        else:
            logger.exception('Unhandled error on %s %s', request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder(
                build_error_body(
                    code=code,
                    message=message,
                    request_id=get_request_id(),
                )
            ),
        )

    application.include_router(health.router)
    application.include_router(algorithms.router)
    application.include_router(datasets.router)
    # The static job paths (``/jobs/compare``, ``/jobs/artefacts.zip``) are
    # registered before the UUID path parameter so they are never captured.
    application.include_router(jobs.router)
    application.include_router(jobs_lifecycle.router)

    @application.get('/', summary='Service banner', tags=['health'])
    async def root() -> dict:
        """Identify the service and its version."""
        return {'service': 'skl-backend', 'version': '0.1.0'}

    return application


app = create_app()


def main() -> None:
    """Entry point for the `backend` console script."""
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)


if __name__ == '__main__':
    main()
