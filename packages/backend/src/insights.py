"""Dataset profiling for the insights panel (FEATURE-16).

The profile is computed once at upload time and stored next to the dataset, so
it can be re-read later without re-parsing the CSV. It is a diagnostic aid:
nothing in this module blocks a training job.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

__all__ = [
    'CATEGORICAL_MAX_CARDINALITY',
    'HIGH_MISSING_RATE',
    'WARNING_ACTIONS',
    'profile_dataframe',
    'preview_rows',
]

#: A string column with more distinct values than this is inferred as ``text``.
CATEGORICAL_MAX_CARDINALITY = 50

#: A column whose missing rate reaches this threshold raises ``HIGH_MISSING_RATE``.
HIGH_MISSING_RATE = 0.3

#: How many of the most frequent values are reported per column.
TOP_VALUES_LIMIT = 5

BOOLEAN_VALUES = frozenset({'true', 'false', 'yes', 'no', '0', '1'})

_DATE_PATTERNS = (
    re.compile(r'^\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?'),
    re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4}$'),
)

#: The suggested action shown next to every warning code.
WARNING_ACTIONS = {
    'CONSTANT_COLUMN': 'Remove this column from the feature set.',
    'HIGH_MISSING_RATE': 'Impute or drop this column.',
    'ALL_VALUES_MISSING': 'This column carries no information.',
    'HIGH_CARDINALITY': 'Looks like an identifier — keep it out of the features.',
    'POSSIBLE_ID_COLUMN': 'Every row has a unique value — probably an ID.',
    'MIXED_TYPES': 'Convert this column to a single type before using it.',
    'TEXT_COLUMN': 'Not usable as a numeric feature.',
    'SINGLE_VALUE_LABEL': 'Add more labelled rows.',
}


def _is_date_like(value: str) -> bool:
    """Whether a value looks like a date or timestamp."""
    return any(pattern.match(value) for pattern in _DATE_PATTERNS)


def _numeric_values(series: pd.Series) -> pd.Series:
    """Return the numeric view of a string series (``NaN`` where not numeric)."""
    return pd.to_numeric(series, errors='coerce')


def _clean_number(value: Any) -> float | None:
    """Return a JSON-friendly float, or ``None`` for a missing value."""
    if value is None:
        return None
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    if number != number:  # NaN
        return None
    return number


def _infer_type(series: pd.Series, values: pd.Series) -> tuple[str, bool]:
    """Infer the column type and whether the column mixes numbers and text.

    Returns ``(inferred_type, mixed_types)``.
    """
    if len(values) == 0:
        return 'categorical', False

    lowered = values.astype(str).str.strip().str.lower()
    numeric = _numeric_values(values)
    numeric_count = int(numeric.notna().sum())

    if numeric_count == len(values):
        return 'numeric', False
    if numeric_count > 0:
        # Numbers and non-numbers in the same column.
        return 'categorical', True
    if set(lowered) <= BOOLEAN_VALUES:
        return 'boolean', False
    if all(_is_date_like(str(value)) for value in values):
        return 'datetime', False
    if values.nunique(dropna=True) <= CATEGORICAL_MAX_CARDINALITY:
        return 'categorical', False
    return 'text', False


def _numeric_summary(series: pd.Series) -> dict[str, float | None] | None:
    """Build the numeric summary of a column, or ``None`` when it has no numbers."""
    numeric = _numeric_values(series).dropna()
    if len(numeric) == 0:
        return None
    return {
        'min': _clean_number(numeric.min()),
        'max': _clean_number(numeric.max()),
        'mean': _clean_number(numeric.mean()),
        'std': _clean_number(numeric.std()),
        'median': _clean_number(numeric.median()),
    }


def _top_values(series: pd.Series, inferred_type: str) -> list[dict[str, Any]]:
    """Return the most frequent values of a non-numeric column."""
    if inferred_type == 'numeric':
        return []
    counts = series.dropna().astype(str).value_counts().head(TOP_VALUES_LIMIT)
    return [{'value': str(value), 'count': int(count)} for value, count in counts.items()]


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Compute the per-column profile and the dataset-level warnings.

    ``df`` must hold strings (``dtype=str``) so the inferred types follow the
    documented rules rather than the pandas dtype heuristics.
    """
    row_count = int(len(df))
    columns: list[dict[str, Any]] = []
    total_missing = 0
    candidate_label_columns: list[dict[str, Any]] = []

    for name in df.columns:
        series = df[name]
        missing_count = int(series.isna().sum())
        missing_rate = (missing_count / row_count) if row_count else 0.0
        values = series.dropna()
        unique_count = int(values.nunique())
        total_missing += missing_count

        inferred_type, mixed_types = _infer_type(series, values)
        warnings: list[str] = []

        if missing_count == row_count and row_count > 0:
            warnings.append('ALL_VALUES_MISSING')
        if missing_rate >= HIGH_MISSING_RATE and missing_count != row_count:
            warnings.append('HIGH_MISSING_RATE')
        if unique_count == 1:
            warnings.append('CONSTANT_COLUMN')
        if row_count and unique_count == row_count:
            warnings.append('POSSIBLE_ID_COLUMN')
        if unique_count > 50 and row_count and unique_count > row_count * 0.5:
            warnings.append('HIGH_CARDINALITY')
        if mixed_types:
            warnings.append('MIXED_TYPES')
        if inferred_type == 'text':
            warnings.append('TEXT_COLUMN')

        column = {
            'name': str(name),
            'inferredType': inferred_type,
            'missingCount': missing_count,
            'missingRate': missing_rate,
            'uniqueCount': unique_count,
            'numeric': _numeric_summary(series),
            'topValues': _top_values(series, inferred_type),
            'warnings': warnings,
        }
        columns.append(column)

        if inferred_type in ('categorical', 'boolean') and unique_count > 0:
            candidate_label_columns.append(column)

    if candidate_label_columns and all(
        column['uniqueCount'] == 1 for column in candidate_label_columns
    ):
        for column in candidate_label_columns:
            if 'SINGLE_VALUE_LABEL' not in column['warnings']:
                column['warnings'].append('SINGLE_VALUE_LABEL')

    dataset_warnings: list[dict[str, Any]] = []
    for column in columns:
        for code in column['warnings']:
            if code == 'SINGLE_VALUE_LABEL':
                continue
            dataset_warnings.append(
                {
                    'code': code,
                    'column': column['name'],
                    'message': f'{column["name"]}: {_warning_message(code, column)}',
                }
            )
    if candidate_label_columns and all(
        column['uniqueCount'] == 1 for column in candidate_label_columns
    ):
        dataset_warnings.append(
            {
                'code': 'SINGLE_VALUE_LABEL',
                'column': None,
                'message': 'Every candidate label column holds a single value.',
            }
        )

    return {
        'rowCount': row_count,
        'columnCount': len(columns),
        'totalMissing': total_missing,
        'columns': columns,
        'warnings': dataset_warnings,
    }


def _warning_message(code: str, column: dict[str, Any]) -> str:
    """Build the human-readable message of a per-column warning."""
    name = column['name']
    if code == 'CONSTANT_COLUMN':
        return f'{name} holds a single value for every row.'
    if code == 'HIGH_MISSING_RATE':
        return f'{name} is missing in {column["missingRate"]:.0%} of the rows.'
    if code == 'ALL_VALUES_MISSING':
        return f'{name} is empty in every row.'
    if code == 'HIGH_CARDINALITY':
        return f'{name} has {column["uniqueCount"]} distinct values.'
    if code == 'POSSIBLE_ID_COLUMN':
        return f'{name} is unique in every row — it looks like an identifier.'
    if code == 'MIXED_TYPES':
        return f'{name} mixes numbers and non-numbers.'
    if code == 'TEXT_COLUMN':
        return f'{name} is free text and cannot be used as a numeric feature.'
    return f'{name} raised {code}.'


def preview_rows(df: pd.DataFrame, offset: int, limit: int) -> list[list[str | None]]:
    """Return raw preview rows, positionally aligned with the column order.

    A missing cell is ``None``; every other cell is its source text.
    """
    if offset >= len(df):
        return []
    window = df.iloc[offset : offset + limit]
    rows: list[list[str | None]] = []
    for _, row in window.iterrows():
        cells: list[str | None] = []
        for value in row:
            if value is None or (isinstance(value, float) and value != value):
                cells.append(None)
            else:
                cells.append(str(value))
        rows.append(cells)
    return rows
