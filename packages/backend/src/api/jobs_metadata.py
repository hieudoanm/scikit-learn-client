"""Job metadata editing: name and tag normalisation (FEATURE-13).

Only the human-readable metadata of a job is editable; the training
configuration is immutable once the job has been submitted.
"""

from __future__ import annotations

from src.api.deps import raise_http

__all__ = ['MAX_NAME_LENGTH', 'MAX_TAGS', 'MAX_TAG_LENGTH', 'normalise_name', 'normalise_tags']

MAX_NAME_LENGTH = 120
MAX_TAGS = 10
MAX_TAG_LENGTH = 32

#: Characters a tag may contain besides letters and digits.
TAG_PUNCTUATION = frozenset({'-', '_', '.', ' '})


def normalise_name(name: str | None) -> str | None:
    """Trim the job name; a blank name is stored as ``None``."""
    if name is None:
        return None
    trimmed = name.strip()
    if not trimmed:
        return None
    if len(trimmed) > MAX_NAME_LENGTH:
        raise_http(
            400,
            'VALIDATION_ERROR',
            f'name must be at most {MAX_NAME_LENGTH} characters',
            details={'field': 'name', 'limit': MAX_NAME_LENGTH},
        )
    return trimmed


def _valid_tag_char(char: str) -> bool:
    """Whether a character may appear in a tag."""
    return char.isalnum() or char in TAG_PUNCTUATION


def normalise_tags(tags: list[str] | None) -> list[str]:
    """Lower-case, trim, validate, and de-duplicate a list of tags."""
    if not tags:
        return []
    normalised: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        if not isinstance(raw, str):
            raise_http(
                400,
                'VALIDATION_ERROR',
                'tags must be strings',
                details={'field': 'tags'},
            )
        tag = raw.strip().lower()
        if not tag:
            raise_http(
                400,
                'VALIDATION_ERROR',
                'A tag must not be empty',
                details={'field': 'tags'},
            )
        if len(tag) > MAX_TAG_LENGTH:
            raise_http(
                400,
                'VALIDATION_ERROR',
                f'A tag must be at most {MAX_TAG_LENGTH} characters',
                details={'field': 'tags', 'limit': MAX_TAG_LENGTH},
            )
        if not all(_valid_tag_char(char) for char in tag):
            raise_http(
                400,
                'VALIDATION_ERROR',
                f'Invalid characters in tag: {raw}',
                details={'field': 'tags'},
            )
        if tag in seen:
            continue
        seen.add(tag)
        normalised.append(tag)
    if len(normalised) > MAX_TAGS:
        raise_http(
            400,
            'VALIDATION_ERROR',
            f'A job may carry at most {MAX_TAGS} tags',
            details={'field': 'tags', 'limit': MAX_TAGS},
        )
    return normalised
