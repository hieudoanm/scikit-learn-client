"""Upload validation for dataset CSVs.

Three independent checks guard the upload endpoint (Phase 12, upload
hardening):

1. the **file extension** must be ``.csv``;
2. the **declared content type** must be one a browser sends for a CSV;
3. the **sniffed bytes** must not belong to a known binary format, and the
   payload must decode as UTF-8 and parse as delimited text.

A payload that is binary (or HTML) is rejected with ``UNSUPPORTED_FILE_TYPE``;
bytes that cannot be decoded or parsed are rejected with ``MALFORMED_CSV``; a
header with an empty or duplicated column name is rejected with
``VALIDATION_ERROR`` and ``details.field = "columns"``.
"""

from __future__ import annotations

import csv
import io

from src.api.deps import raise_http

__all__ = [
    'ALLOWED_CONTENT_TYPES',
    'BINARY_SIGNATURES',
    'MAX_COLUMNS',
    'MAX_MISSING_COLUMN_RATE',
    'parse_csv',
    'validate_content_type',
    'validate_extension',
    'sniff_binary',
]

#: Content types a browser (or curl) may legitimately declare for a CSV.
ALLOWED_CONTENT_TYPES = frozenset(
    {
        '',
        'application/csv',
        'application/excel',
        'application/octet-stream',
        'application/vnd.ms-excel',
        'application/x-csv',
        'text/comma-separated-values',
        'text/csv',
        'text/plain',
    }
)

#: Magic byte signatures of payloads that are never a CSV.
BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b'PK\x03\x04', 'a ZIP archive'),
    (b'PK\x05\x06', 'a ZIP archive'),
    (b'PK\x07\x08', 'a ZIP archive'),
    (b'\x1f\x8b', 'a gzip stream'),
    (b'\x1f\x9d', 'a compressed archive'),
    (b'BZh', 'a bzip2 archive'),
    (b'\xfd7zXZ\x00', 'an xz archive'),
    (b'Rar!\x1a\x07', 'a RAR archive'),
    (b'\x89PNG\r\n\x1a\n', 'a PNG image'),
    (b'\xff\xd8\xff', 'a JPEG image'),
    (b'GIF87a', 'a GIF image'),
    (b'GIF89a', 'a GIF image'),
    (b'%PDF-', 'a PDF document'),
    (b'\x7fELF', 'an ELF binary'),
    (b'\xca\xfe\xba\xbe', 'a Java class file'),
    (b'MZ', 'a Windows executable'),
)

#: Prefixes of documents that are text but still not a CSV.
HTML_PREFIXES = (b'<!doctype html', b'<html', b'<?xml', b'<svg', b'<script')

#: A UTF-16 byte-order mark: decodable as UTF-16, never as UTF-8.
UTF16_BOMS = (b'\xff\xfe', b'\xfe\xff')

#: How many leading bytes are inspected for a NUL byte.
SNIFF_WINDOW = 4096

#: Upper bound on the number of columns a dataset may carry.
MAX_COLUMNS = 512

#: A column whose cells are empty in at least this fraction of rows carries no
#: signal for training and is rejected up front. The check is deliberately
#: generous: a genuinely sparse feature is unusual, whereas a trailing comma or
#: an export artefact produces a fully empty column.
MAX_MISSING_COLUMN_RATE = 1.0


def _empty_column_names(header: list[str], data_rows: list[list[str]]) -> list[str]:
    """Return the columns that are empty in every data row.

    ``csv.reader`` pads short rows, so a row that stops early is treated as
    empty for the missing trailing cells. Only columns that are empty in *all*
    rows are returned; a partially populated column is left to the insights
    view (FEATURE-16) to flag.
    """
    empty: list[str] = []
    for index, name in enumerate(header):
        if all(index >= len(row) or row[index].strip() == '' for row in data_rows):
            empty.append(name)
    return empty


def validate_extension(filename: str | None) -> str:
    """Check the file extension and return the file name."""
    name = filename or 'upload.csv'
    if not name.lower().endswith('.csv'):
        raise_http(
            400,
            'UNSUPPORTED_FILE_TYPE',
            'Only .csv uploads are supported',
            details={'filename': name, 'expectedExtension': '.csv'},
        )
    return name


def validate_content_type(content_type: str | None) -> str:
    """Check the declared content type of the upload."""
    declared = (content_type or '').split(';')[0].strip().lower()
    if declared not in ALLOWED_CONTENT_TYPES:
        raise_http(
            400,
            'UNSUPPORTED_FILE_TYPE',
            f'Unsupported content type: {declared or "unknown"}',
            details={'contentType': declared, 'allowed': sorted(ALLOWED_CONTENT_TYPES)},
        )
    return declared


def sniff_binary(data: bytes) -> str | None:
    """Return a label when the payload looks like a known binary format.

    ``None`` means the payload does not carry a binary signature; the caller
    still decodes it and reports a malformed CSV when that fails.
    """
    for signature, label in BINARY_SIGNATURES:
        if data.startswith(signature):
            return label

    for bom in UTF16_BOMS:
        if data.startswith(bom):
            return 'a UTF-16 encoded file'

    lowered = data[:SNIFF_WINDOW].lstrip().lower()
    for prefix in HTML_PREFIXES:
        if lowered.startswith(prefix):
            return 'an HTML document'

    if b'\x00' in data[:SNIFF_WINDOW]:
        return 'a binary file'

    return None


def parse_csv(text: str) -> tuple[list[str], int]:
    """Parse the CSV text and return ``(header, data_row_count)``.

    Raises ``400 MALFORMED_CSV`` when the text is not delimited text, and
    ``400 VALIDATION_ERROR`` when the header is empty, holds an empty or
    duplicated column name, or the file holds no data row.
    """
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except csv.Error as exc:
        raise_http(400, 'MALFORMED_CSV', f'File cannot be parsed as CSV: {exc}')

    if not rows:
        raise_http(400, 'MALFORMED_CSV', 'File cannot be parsed as CSV: no content')

    header = [cell.strip() for cell in rows[0]]
    data_rows = [row for row in rows[1:] if any(cell.strip() for cell in row)]

    if not header or not any(header):
        raise_http(400, 'VALIDATION_ERROR', 'CSV has no header row')

    if len(header) < 2:
        raise_http(
            400,
            'VALIDATION_ERROR',
            'CSV must have a header and at least one data column',
            details={'field': 'columns', 'count': len(header)},
        )

    if len(header) > MAX_COLUMNS:
        raise_http(
            400,
            'VALIDATION_ERROR',
            f'CSV must have at most {MAX_COLUMNS} columns',
            details={'field': 'columns', 'limit': MAX_COLUMNS, 'count': len(header)},
        )

    for index, name in enumerate(header):
        if not name:
            raise_http(
                400,
                'VALIDATION_ERROR',
                f'Column {index + 1} has an empty name',
                details={'field': 'columns', 'index': index},
            )

    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise_http(
            400,
            'VALIDATION_ERROR',
            f'Duplicate column name(s): {", ".join(duplicates)}',
            details={'field': 'columns', 'duplicates': duplicates},
        )

    if not data_rows:
        raise_http(
            400,
            'VALIDATION_ERROR',
            'CSV has a header but no data rows',
            details={'field': 'columns'},
        )

    empty_columns = _empty_column_names(header, data_rows)
    if len(empty_columns) >= len(header):
        raise_http(
            400,
            'VALIDATION_ERROR',
            'Every column is empty; the CSV has no usable data',
            details={'field': 'columns', 'columns': empty_columns},
        )
    if empty_columns:
        raise_http(
            400,
            'VALIDATION_ERROR',
            f'Column(s) empty in every row: {", ".join(empty_columns)}',
            details={'field': 'columns', 'emptyColumns': empty_columns},
        )

    return header, len(data_rows)
