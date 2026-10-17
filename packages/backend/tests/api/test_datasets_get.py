"""Unit tests for the dataset retrieval endpoint (src/api/datasets.py)."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.api.datasets import get_dataset
from src.api.schemas import DatasetResponse


@pytest.mark.asyncio
async def test_get_dataset_success(mock_session):
    """Test successful dataset retrieval."""
    dataset_id = uuid.UUID('12345678-1234-5678-1234-567812345678')

    # Mock dataset object
    mock_dataset = MagicMock()
    mock_dataset.id = dataset_id
    mock_dataset.sha256 = 'abc123'
    mock_dataset.original_filename = 'test.csv'
    mock_dataset.byte_size = 100
    mock_dataset.columns = [
        {'name': 'col1', 'inferredDtype': 'int64', 'nullCount': 0, 'sampleValues': [1, 2]}
    ]
    mock_dataset.row_count = 10
    mock_dataset.created_at = datetime(2024, 1, 1, 0, 0, 0)

    # Mock database query result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=mock_dataset)
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await get_dataset(mock_session, dataset_id)

    assert isinstance(result, DatasetResponse)
    assert str(result.datasetId) == str(dataset_id)
    assert result.sha256 == 'abc123'
    assert result.filename == 'test.csv'
    assert result.byteSize == 100
    assert result.rowCount == 10
    assert result.createdAt == datetime(2024, 1, 1, 0, 0, 0)


@pytest.mark.asyncio
async def test_get_dataset_not_found(mock_session):
    """Test dataset retrieval when dataset doesn't exist."""
    dataset_id = uuid.UUID('12345678-1234-5678-1234-567812345678')

    # Mock database query result - None
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(Exception) as exc_info:
        await get_dataset(mock_session, dataset_id)
    assert 'DATASET_NOT_FOUND' in str(exc_info.value)
    assert str(dataset_id) in str(exc_info.value)
