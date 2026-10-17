"""Unit tests for src/api/deps.py."""

import pytest
from fastapi import HTTPException
from src.api.deps import build_error_body, raise_http


def test_build_error_body_minimal():
    """build_error_body omits details when not supplied."""
    body = build_error_body('NOT_FOUND', 'Resource not found', 'req-1')
    assert body == {
        'error': {'code': 'NOT_FOUND', 'message': 'Resource not found', 'requestId': 'req-1'}
    }


def test_build_error_body_with_details():
    """build_error_body includes details when supplied."""
    body = build_error_body('BAD', 'Nope', 'req-2', details={'missing': ['a']})
    assert body['error']['details'] == {'missing': ['a']}


def test_raise_http_basic():
    """Test raise_http with basic parameters."""
    with pytest.raises(HTTPException) as exc_info:
        raise_http(400, 'VALIDATION_ERROR', 'Invalid input')

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail['error']['code'] == 'VALIDATION_ERROR'
    assert exc_info.value.detail['error']['message'] == 'Invalid input'
    assert 'requestId' in exc_info.value.detail['error']
    assert 'details' not in exc_info.value.detail['error']


def test_raise_http_with_details():
    """Test raise_http with details parameter."""
    with pytest.raises(HTTPException) as exc_info:
        raise_http(
            400,
            'INVALID_COLUMN_MAP',
            'Missing columns',
            details={'missing': ['col1', 'col2']},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail['error'] == {
        'code': 'INVALID_COLUMN_MAP',
        'message': 'Missing columns',
        'requestId': exc_info.value.detail['error']['requestId'],
        'details': {'missing': ['col1', 'col2']},
    }


def test_raise_http_with_none_details():
    """Test raise_http with None details (should not include details in response)."""
    with pytest.raises(HTTPException) as exc_info:
        raise_http(404, 'NOT_FOUND', 'Resource not found', details=None)

    assert exc_info.value.status_code == 404
    assert 'details' not in exc_info.value.detail['error']


def test_raise_http_empty_details():
    """Test raise_http with empty dict details."""
    with pytest.raises(HTTPException) as exc_info:
        raise_http(500, 'SERVER_ERROR', 'Internal error', details={})

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail['error']['details'] == {}
