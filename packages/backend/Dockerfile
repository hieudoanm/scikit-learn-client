# Build from the repository root so the root `.dockerignore` applies and the
# lockfile is in context:
#
#   docker build -f packages/backend/Dockerfile -t skl-backend .
#
# The runtime image runs the API with `uvicorn main:app` (see `main.py`).
# See docs/11-deployment/local-development.md.

# --- Build stage: resolve and install the locked virtualenv ----------------
FROM python:3.15.1-slim AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.14 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app

# Only the manifest, lockfile, and readme are needed to resolve dependencies,
# so this layer is cached until one of them changes.
COPY packages/backend/pyproject.toml packages/backend/uv.lock packages/backend/README.md ./
RUN uv sync --frozen --no-dev --extra onnx

# --- Runtime stage: copy the venv and application code --------------------
FROM python:3.15.1-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SKL_DATA_DIR=/var/lib/skl \
    PATH=/opt/venv/bin:$PATH

RUN groupadd --system --gid 1001 skl \
    && useradd --system --uid 1001 --gid skl --home-dir /app --shell /usr/sbin/nologin skl

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY packages/backend/main.py ./
COPY packages/backend/src ./src
COPY packages/backend/migrations ./migrations
COPY packages/backend/alembic.ini ./

RUN mkdir -p /var/lib/skl/datasets /var/lib/skl/artefacts \
    && chown -R skl:skl /var/lib/skl /app

USER skl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=5 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
