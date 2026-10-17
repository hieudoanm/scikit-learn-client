"""Parsing of comma-separated id lists used by the comparison and bundle endpoints."""

from __future__ import annotations

import uuid

from src.api.deps import raise_http

__all__ = ['parse_id_list']


def parse_id_list(
    raw: str | None,
    *,
    minimum: int,
    maximum: int,
    field: str = 'ids',
) -> list[uuid.UUID]:
    """Parse a comma-separated list of UUIDs, collapsing duplicates.

    Raises ``400 VALIDATION_ERROR`` when the list is empty, too long, or holds a
    malformed UUID.
    """
    if raw is None or not raw.strip():
        raise_http(
            400,
            'VALIDATION_ERROR',
            f'{field} must contain between {minimum} and {maximum} job ids',
            details={'field': field, 'min': minimum, 'max': maximum},
        )
    parts = [part.strip() for part in raw.split(',') if part.strip()]
    ids: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for part in parts:
        try:
            parsed = uuid.UUID(part)
        except ValueError:
            raise_http(
                400,
                'VALIDATION_ERROR',
                f'Malformed job id: {part}',
                details={'field': field},
            )
        if parsed not in seen:
            seen.add(parsed)
            ids.append(parsed)
    if len(ids) < minimum:
        raise_http(
            400,
            'VALIDATION_ERROR',
            f'{field} must contain at least {minimum} job ids',
            details={'field': field, 'min': minimum},
        )
    if len(ids) > maximum:
        raise_http(
            400,
            'VALIDATION_ERROR',
            f'{field} must contain at most {maximum} job ids',
            details={'field': field, 'max': maximum},
        )
    return ids
