"""Unit tests for the dataset insights (FEATURE-16, scenario 35)."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from src.api.datasets import get_dataset_insights
from src.insights import preview_rows, profile_dataframe


def _frame(rows, columns):
    """Build a string-typed frame."""
    return pd.DataFrame(rows, columns=columns, dtype=str)


def test_profile_reports_every_column():
    """The profile holds one entry per column."""
    frame = _frame([['1.0', 'a'], ['2.0', 'b']], ['feature1', 'feature2'])
    profile = profile_dataframe(frame)

    assert profile['rowCount'] == 2
    assert profile['columnCount'] == 2
    assert [column['name'] for column in profile['columns']] == ['feature1', 'feature2']
    assert profile['columns'][0]['inferredType'] == 'numeric'
    assert profile['columns'][1]['inferredType'] == 'categorical'


def test_profile_summarises_a_numeric_column():
    """A numeric column carries min/max/mean/std/median."""
    frame = _frame([['1'], ['2'], ['3']], ['feature1'])
    column = profile_dataframe(frame)['columns'][0]

    assert column['numeric']['min'] == 1.0
    assert column['numeric']['max'] == 3.0
    assert column['numeric']['mean'] == 2.0
    assert column['numeric']['median'] == 2.0
    assert column['topValues'] == []


def test_profile_reports_the_top_values_of_a_categorical_column():
    """A categorical column lists its most frequent values."""
    frame = _frame([['a'], ['a'], ['b']], ['feature1'])
    column = profile_dataframe(frame)['columns'][0]

    assert column['topValues'][0] == {'value': 'a', 'count': 2}
    assert column['topValues'][1] == {'value': 'b', 'count': 1}


def test_profile_flags_a_constant_column():
    """A column holding a single value raises CONSTANT_COLUMN."""
    frame = _frame([['1', 'x'], ['2', 'x']], ['feature1', 'constant'])
    profile = profile_dataframe(frame)

    constant = next(column for column in profile['columns'] if column['name'] == 'constant')
    assert 'CONSTANT_COLUMN' in constant['warnings']
    assert any(warning['code'] == 'CONSTANT_COLUMN' for warning in profile['warnings'])


def test_profile_flags_a_possible_id_column():
    """A column whose distinct count equals the row count raises POSSIBLE_ID_COLUMN."""
    frame = _frame([['1', 'a'], ['2', 'b'], ['3', 'c']], ['feature1', 'row_id'])
    profile = profile_dataframe(frame)

    identifier = next(column for column in profile['columns'] if column['name'] == 'row_id')
    assert 'POSSIBLE_ID_COLUMN' in identifier['warnings']
    assert any(warning['code'] == 'POSSIBLE_ID_COLUMN' for warning in profile['warnings'])


def test_profile_flags_a_high_missing_rate():
    """A column missing in at least 30% of the rows raises HIGH_MISSING_RATE."""
    frame = _frame(
        [['1', 'x'], ['2', None], ['3', None], ['4', None]],
        ['feature1', 'sparse'],
    )
    profile = profile_dataframe(frame)

    sparse = next(column for column in profile['columns'] if column['name'] == 'sparse')
    assert 'HIGH_MISSING_RATE' in sparse['warnings']
    assert sparse['missingCount'] == 3


def test_profile_flags_an_entirely_missing_column():
    """A column that is empty in every row raises ALL_VALUES_MISSING."""
    frame = _frame([['1', None], ['2', None]], ['feature1', 'empty'])
    profile = profile_dataframe(frame)

    empty = next(column for column in profile['columns'] if column['name'] == 'empty')
    assert 'ALL_VALUES_MISSING' in empty['warnings']
    assert profile['totalMissing'] == 2


def test_profile_flags_mixed_types():
    """A column mixing numbers and text raises MIXED_TYPES."""
    frame = _frame([['1'], ['n/a'], ['3']], ['feature1'])
    column = profile_dataframe(frame)['columns'][0]

    assert column['inferredType'] == 'categorical'
    assert 'MIXED_TYPES' in column['warnings']


def test_profile_detects_a_boolean_column():
    """A column holding only boolean-ish words is inferred as boolean."""
    frame = _frame([['yes'], ['no'], ['yes']], ['flag'])
    column = profile_dataframe(frame)['columns'][0]

    assert column['inferredType'] == 'boolean'


def test_profile_detects_a_datetime_column():
    """A column holding ISO dates is inferred as datetime."""
    frame = _frame([['2024-01-01'], ['2024-02-01']], ['when'])
    column = profile_dataframe(frame)['columns'][0]

    assert column['inferredType'] == 'datetime'


def test_profile_flags_a_single_value_label():
    """When every candidate label column holds one value, SINGLE_VALUE_LABEL is raised."""
    frame = _frame([['1', 'yes'], ['2', 'yes']], ['feature1', 'label'])
    profile = profile_dataframe(frame)

    assert any(warning['code'] == 'SINGLE_VALUE_LABEL' for warning in profile['warnings'])


def test_preview_rows_are_positional():
    """The preview rows follow the column order and keep the source text."""
    frame = _frame([['1', 'a'], ['2', 'b'], ['3', 'c']], ['feature1', 'feature2'])
    rows = preview_rows(frame, 0, 2)

    assert rows == [['1', 'a'], ['2', 'b']]


def test_preview_rows_honour_the_offset():
    """The offset skips the leading rows."""
    frame = _frame([['1'], ['2'], ['3']], ['feature1'])
    assert preview_rows(frame, 1, 1) == [['2']]


def test_preview_rows_past_the_end_are_empty():
    """An offset past the last row yields no rows."""
    frame = _frame([['1']], ['feature1'])
    assert preview_rows(frame, 5, 5) == []


def _dataset_row(dataset_id):
    """Build a mock dataset row."""
    dataset = MagicMock()
    dataset.id = dataset_id
    dataset.sha256 = 'sha'
    dataset.original_filename = 'sample.csv'
    dataset.byte_size = 100
    dataset.row_count = 3
    dataset.columns = [{'name': 'feature1'}, {'name': 'label'}]
    dataset.storage_key = f'datasets/{dataset_id}/data.csv'
    return dataset


@pytest.mark.asyncio
async def test_get_dataset_insights_rejects_an_out_of_range_preview_limit(mock_session):
    """`previewLimit` outside 1–100 is a validation error."""
    with pytest.raises(Exception) as exc_info:
        await get_dataset_insights(mock_session, uuid.uuid4(), preview_limit=101)
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'previewLimit' in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_dataset_insights_of_an_unknown_dataset(mock_session):
    """An unknown dataset answers 404 DATASET_NOT_FOUND."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(Exception) as exc_info:
        await get_dataset_insights(mock_session, uuid.uuid4())
    assert 'DATASET_NOT_FOUND' in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_dataset_insights_returns_the_stored_profile(mock_session, tmp_path, monkeypatch):
    """The stored profile is returned with a preview of the source rows."""
    dataset_id = uuid.uuid4()
    dataset = _dataset_row(dataset_id)

    source = tmp_path / 'data.csv'
    source.write_text('feature1,label\n1,0\n2,1\n3,0\n')
    monkeypatch.setattr('src.api.datasets.dataset_path', lambda key: source)

    insight = MagicMock()
    insight.row_count = 3
    insight.column_count = 2
    insight.total_missing = 0
    insight.columns = [
        {'name': 'feature1', 'inferredType': 'numeric'},
        {'name': 'label', 'inferredType': 'categorical'},
    ]
    insight.warnings = []

    dataset_result = MagicMock()
    dataset_result.scalar_one_or_none = MagicMock(return_value=dataset)
    insight_result = MagicMock()
    insight_result.scalar_one_or_none = MagicMock(return_value=insight)

    mock_session.execute = AsyncMock(side_effect=[dataset_result, insight_result])

    response = await get_dataset_insights(mock_session, dataset_id)

    assert response.datasetId == dataset_id
    assert response.rowCount == 3
    assert response.columnCount == 2
    assert response.totalMissingCells == 0
    assert len(response.columns) == 2
    assert response.preview.rows == [['1', '0'], ['2', '1'], ['3', '0']]
    assert response.preview.columns == ['feature1', 'label']
