"""Dataset upload, metadata, and insights endpoints."""

from __future__ import annotations

import uuid
from io import StringIO
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Query, UploadFile, status
from sqlalchemy import insert, select

from src.api.datasets_validation import (
    parse_csv,
    sniff_binary,
    validate_content_type,
    validate_extension,
)
from src.api.deps import DBSession, raise_http
from src.api.schemas import (
    ColumnPreview,
    DatasetInsightsResponse,
    DatasetPreview,
    DatasetResponse,
)
from src.db.models import Dataset, DatasetInsight
from src.db.utils import utc_now
from src.insights import preview_rows, profile_dataframe
from src.storage.filesystem import (
    MAX_UPLOAD_BYTES,
    dataset_path,
    ensure_dirs,
    write_dataset,
)

router = APIRouter(tags=['datasets'])

#: Bounds of the ``previewLimit`` query parameter of the insights endpoint.
MIN_PREVIEW_LIMIT = 1
MAX_PREVIEW_LIMIT = 100
DEFAULT_PREVIEW_LIMIT = 10


def _decode_csv(file_bytes: bytes) -> str:
    """Decode the upload as UTF-8, rejecting a payload that is not UTF-8."""
    try:
        return file_bytes.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise_http(
            400,
            'MALFORMED_CSV',
            f'File is not valid UTF-8: {exc.reason}',
        )
        raise  # pragma: no cover - raise_http always raises


@router.post(
    '/api/v1/datasets',
    summary='Upload CSV dataset',
    status_code=status.HTTP_201_CREATED,
    response_model=DatasetResponse,
)
async def upload_dataset(session: DBSession, file: UploadFile = File(...)) -> DatasetResponse:
    """Accept a CSV upload, persist bytes, hash, and infer column preview.

    The extension, the declared content type, and the sniffed bytes are all
    checked before anything is written to the volume.
    """
    ensure_dirs()

    filename = validate_extension(file.filename)
    validate_content_type(file.content_type)

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise_http(
            413,
            'UPLOAD_TOO_LARGE',
            f'File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB',
        )
    if len(file_bytes) == 0:
        raise_http(400, 'VALIDATION_ERROR', 'Empty file uploaded')

    binary = sniff_binary(file_bytes)
    if binary is not None:
        raise_http(
            400,
            'UNSUPPORTED_FILE_TYPE',
            f'File appears to be {binary}, not a CSV',
            details={'filename': filename},
        )

    text = _decode_csv(file_bytes)
    if not text.strip():
        raise_http(400, 'VALIDATION_ERROR', 'CSV has no content')

    header, data_row_count = parse_csv(text)

    # Persist
    dataset_id, storage_key, sha256, byte_size = write_dataset(file_bytes, filename)

    # Column preview (full read: dtype and missing counts must be accurate)
    df_all = pd.read_csv(StringIO(text))
    columns = []
    for col in df_all.columns:
        series = df_all[col]
        columns.append(
            {
                'name': col,
                'inferredDtype': str(series.dtype),
                'nullCount': int(series.isna().sum()),
                'sampleValues': [None if pd.isna(value) else value for value in series.head(5)],
            }
        )
    row_count = len(df_all)

    # Insights profile: computed once, stored with the dataset (FEATURE-16).
    profile = profile_dataframe(pd.read_csv(StringIO(text), dtype=str))

    now = utc_now()
    stmt = insert(Dataset).values(
        id=dataset_id,
        original_filename=filename,
        sha256=sha256,
        byte_size=byte_size,
        storage_key=storage_key,
        columns=columns,
        row_count=row_count,
        created_at=now,
    )
    await session.execute(stmt)

    insight_stmt = insert(DatasetInsight).values(
        dataset_id=dataset_id,
        row_count=profile['rowCount'],
        column_count=profile['columnCount'],
        total_missing=profile['totalMissing'],
        columns=profile['columns'],
        warnings=profile['warnings'],
        created_at=now,
    )
    await session.execute(insight_stmt)
    await session.commit()

    return DatasetResponse(
        datasetId=dataset_id,
        sha256=sha256,
        filename=filename,
        byteSize=byte_size,
        columns=[ColumnPreview(**c) for c in columns],
        rowCount=row_count,
        createdAt=now,
    )


@router.get(
    '/api/v1/datasets/{dataset_id}', summary='Dataset metadata', response_model=DatasetResponse
)
async def get_dataset(session: DBSession, dataset_id: uuid.UUID) -> DatasetResponse:
    """Return dataset metadata and column preview."""
    result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise_http(404, 'DATASET_NOT_FOUND', f'Dataset {dataset_id} not found')
    return DatasetResponse(
        datasetId=dataset.id,
        sha256=dataset.sha256,
        filename=dataset.original_filename,
        byteSize=dataset.byte_size,
        columns=[ColumnPreview(**c) for c in dataset.columns],
        rowCount=dataset.row_count,
        createdAt=dataset.created_at,
    )


@router.get(
    '/api/v1/datasets/{dataset_id}/insights',
    summary='Dataset insights',
    response_model=DatasetInsightsResponse,
)
async def get_dataset_insights(
    session: DBSession,
    dataset_id: uuid.UUID,
    preview_limit: Annotated[int, Query(alias='previewLimit')] = DEFAULT_PREVIEW_LIMIT,
    preview_offset: Annotated[int, Query(alias='previewOffset')] = 0,
) -> DatasetInsightsResponse:
    """Return the per-column profile of a dataset plus a raw preview.

    The profile is computed once at upload time; the preview rows are read
    from the stored CSV on demand.
    """
    if preview_limit < MIN_PREVIEW_LIMIT or preview_limit > MAX_PREVIEW_LIMIT:
        raise_http(
            400,
            'VALIDATION_ERROR',
            f'previewLimit must be between {MIN_PREVIEW_LIMIT} and {MAX_PREVIEW_LIMIT}',
            details={'field': 'previewLimit', 'limit': MAX_PREVIEW_LIMIT},
        )

    if preview_offset < 0:
        raise_http(
            400,
            'VALIDATION_ERROR',
            'previewOffset must not be negative',
            details={'field': 'previewOffset'},
        )

    result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise_http(404, 'DATASET_NOT_FOUND', f'Dataset {dataset_id} not found')

    insight_result = await session.execute(
        select(DatasetInsight).where(DatasetInsight.dataset_id == dataset_id)
    )
    insight = insight_result.scalar_one_or_none()

    if insight is not None:
        columns = list(insight.columns or [])
        warnings = list(insight.warnings or [])
        row_count = insight.row_count
        column_count = insight.column_count
        total_missing = insight.total_missing
    else:
        # A dataset uploaded before the insights feature: profile it on the fly.
        profile = profile_dataframe(_read_dataset_frame(dataset.storage_key))
        columns = profile['columns']
        warnings = profile['warnings']
        row_count = profile['rowCount']
        column_count = profile['columnCount']
        total_missing = profile['totalMissing']

    frame = _read_dataset_frame(dataset.storage_key)
    preview = DatasetPreview(
        offset=preview_offset,
        limit=preview_limit,
        totalRows=row_count,
        columns=[str(name) for name in frame.columns],
        rows=preview_rows(frame, preview_offset, preview_limit),
    )

    cell_count = row_count * column_count
    missing_rate = (total_missing / cell_count) if cell_count else 0.0

    return DatasetInsightsResponse(
        datasetId=dataset.id,
        sha256=dataset.sha256,
        filename=dataset.original_filename,
        rowCount=row_count,
        columnCount=column_count,
        totalMissingCells=total_missing,
        missingRate=missing_rate,
        columns=columns,
        warnings=warnings,
        preview=preview,
    )


def _read_dataset_frame(storage_key: str) -> pd.DataFrame:
    """Read the stored CSV as strings so the preview keeps the source text."""
    path = dataset_path(storage_key)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str)
    except pd.errors.ParserError, UnicodeDecodeError, ValueError:
        return pd.DataFrame()
