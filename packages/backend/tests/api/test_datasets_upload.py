"""Unit tests for the dataset upload endpoint (src/api/datasets.py)."""

import uuid
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile
from src.api.datasets import upload_dataset
from src.api.schemas import DatasetResponse


@pytest.mark.asyncio
async def test_upload_dataset_success(mock_session, sample_csv_bytes, monkeypatch):
    """Test successful dataset upload."""
    # Mock dependencies
    mock_ensure_dirs = MagicMock()
    mock_write_dataset = MagicMock(
        return_value=(
            uuid.UUID('12345678-1234-5678-1234-567812345678'),
            'datasets/123/123.csv',
            'abc123',
            100,
        )
    )
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)
    monkeypatch.setattr('src.api.datasets.write_dataset', mock_write_dataset)

    # Create upload file
    file = UploadFile(filename='test.csv', file=BytesIO(sample_csv_bytes))

    # Call the endpoint
    result = await upload_dataset(mock_session, file)

    # Verify result
    assert isinstance(result, DatasetResponse)
    assert str(result.datasetId) == '12345678-1234-5678-1234-567812345678'
    assert result.sha256 == 'abc123'
    assert result.filename == 'test.csv'
    assert result.byteSize == 100
    assert result.rowCount == 3
    assert len(result.columns) == 3
    assert result.createdAt is not None

    # Verify database operations: the dataset row and its insight profile.
    assert mock_session.execute.call_count == 2
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_upload_dataset_empty_file(mock_session, monkeypatch):
    """Test upload with empty file."""
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    file = UploadFile(filename='empty.csv', file=BytesIO(b''))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'Empty file' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_too_large(mock_session, monkeypatch):
    """Test upload with file exceeding size limit."""
    from src.api.datasets import MAX_UPLOAD_BYTES

    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    large_bytes = b'x' * (MAX_UPLOAD_BYTES + 1)
    file = UploadFile(filename='large.csv', file=BytesIO(large_bytes))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'UPLOAD_TOO_LARGE' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_invalid_utf8(mock_session, monkeypatch):
    """Test upload with non-UTF-8 file."""
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    # Invalid UTF-8 bytes (no BOM): the payload is not decodable text.
    invalid_bytes = b'feature1,label\n\xff\xfe\xfd,1\n'
    file = UploadFile(filename='invalid.csv', file=BytesIO(invalid_bytes))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'MALFORMED_CSV' in str(exc_info.value)
    assert 'UTF-8' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_utf16_bom_rejected(mock_session, monkeypatch):
    """A UTF-16 payload is rejected as an unsupported file type (FEATURE-01)."""
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    utf16_bytes = b'\xff\xfe' + 'feature1,label\n1,0\n'.encode('utf-16-le')
    file = UploadFile(filename='utf16.csv', file=BytesIO(utf16_bytes))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'UNSUPPORTED_FILE_TYPE' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_binary_payload_rejected(mock_session, monkeypatch):
    """A ZIP payload uploaded as .csv is rejected as an unsupported file type."""
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    file = UploadFile(filename='archive.csv', file=BytesIO(b'PK\x03\x04rest-of-the-archive'))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'UNSUPPORTED_FILE_TYPE' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_wrong_extension_rejected(mock_session, monkeypatch):
    """Only .csv uploads are accepted (FEATURE-01)."""
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    file = UploadFile(filename='data.txt', file=BytesIO(b'a,b\n1,2\n'))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'UNSUPPORTED_FILE_TYPE' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_header_only_rejected(mock_session, monkeypatch):
    """A header without data rows is rejected (FEATURE-01)."""
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    file = UploadFile(filename='header.csv', file=BytesIO(b'feature1,label\n'))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'no data rows' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_duplicate_columns_rejected(mock_session, monkeypatch):
    """A duplicated column name is rejected (FEATURE-01)."""
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    file = UploadFile(filename='dupes.csv', file=BytesIO(b'feature1,feature1\n1,2\n'))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'Duplicate column' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_empty_column_rejected(mock_session, monkeypatch):
    """A column that is empty in every row carries no signal (FEATURE-01)."""
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    file = UploadFile(filename='empty.csv', file=BytesIO(b'feature1,blank,label\n1,,0\n2,,1\n'))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'empty in every row' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_all_columns_empty_rejected(mock_session, monkeypatch):
    """A CSV whose every column is empty is rejected as unusable.

    Rows of only commas are dropped as blank first, so this lands on the
    "no data rows" guard rather than the empty-column one.
    """
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    file = UploadFile(filename='blank.csv', file=BytesIO(b'a,b\n,\n,\n'))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'no data rows' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_no_content(mock_session, monkeypatch):
    """Test upload with file that has no content after stripping."""
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    # File with only whitespace
    whitespace_bytes = b'   \n\n   \n'
    file = UploadFile(filename='whitespace.csv', file=BytesIO(whitespace_bytes))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'no content' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_insufficient_columns(mock_session, monkeypatch):
    """Test upload with CSV having only one column."""
    mock_ensure_dirs = MagicMock()
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)

    # CSV with only one column
    single_col_bytes = b'feature1\n1\n2\n3\n'
    file = UploadFile(filename='single.csv', file=BytesIO(single_col_bytes))

    with pytest.raises(Exception) as exc_info:
        await upload_dataset(mock_session, file)
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'header' in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_dataset_no_filename(mock_session, sample_csv_bytes, monkeypatch):
    """Test upload with no filename (should use default)."""
    mock_ensure_dirs = MagicMock()
    mock_write_dataset = MagicMock(
        return_value=(
            uuid.UUID('12345678-1234-5678-1234-567812345678'),
            'datasets/123/123.csv',
            'abc123',
            100,
        )
    )
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)
    monkeypatch.setattr('src.api.datasets.write_dataset', mock_write_dataset)

    file = UploadFile(filename=None, file=BytesIO(sample_csv_bytes))

    result = await upload_dataset(mock_session, file)
    assert isinstance(result, DatasetResponse)
    assert result.filename == 'upload.csv'


@pytest.mark.asyncio
async def test_upload_dataset_column_inference(mock_session, sample_csv_bytes, monkeypatch):
    """Test that column inference works correctly."""
    mock_ensure_dirs = MagicMock()
    mock_write_dataset = MagicMock(
        return_value=(
            uuid.UUID('12345678-1234-5678-1234-567812345678'),
            'datasets/123/123.csv',
            'abc123',
            100,
        )
    )
    monkeypatch.setattr('src.api.datasets.ensure_dirs', mock_ensure_dirs)
    monkeypatch.setattr('src.api.datasets.write_dataset', mock_write_dataset)

    file = UploadFile(filename='test.csv', file=BytesIO(sample_csv_bytes))

    result = await upload_dataset(mock_session, file)

    # Verify column structure
    assert isinstance(result, DatasetResponse)
    assert len(result.columns) == 3
    for col in result.columns:
        assert hasattr(col, 'name')
        assert hasattr(col, 'inferredDtype')
        assert hasattr(col, 'sampleValues')


@pytest.mark.asyncio
async def test_upload_dataset_reports_null_counts(mock_session, monkeypatch):
    """Column previews report missing cells for the semi-supervised UI (FR-081)."""
    csv_bytes = b'feature1,feature2,label\n1.0,2.0,0\n2.0,,1\n3.0,1.0,\n'
    upload = UploadFile(filename='semi.csv', file=BytesIO(csv_bytes))

    monkeypatch.setattr('src.api.datasets.ensure_dirs', MagicMock())
    monkeypatch.setattr(
        'src.api.datasets.write_dataset',
        MagicMock(
            return_value=(
                uuid.UUID('12345678-1234-5678-1234-567812345678'),
                'datasets/1/semi.csv',
                'sha',
                52,
            )
        ),
    )

    result = await upload_dataset(mock_session, upload)

    assert isinstance(result, DatasetResponse)
    columns = {column.name: column for column in result.columns}
    assert columns['feature1'].nullCount == 0
    assert columns['feature2'].nullCount == 1
    assert columns['label'].nullCount == 1
    assert result.rowCount == 3
