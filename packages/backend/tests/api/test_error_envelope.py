"""Tests for the skl error envelope and request-body validation (docs/06 §6)."""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_validation_error_uses_flat_envelope():
    """A malformed request body returns 400 with the flat skl envelope."""
    response = client.post('/api/v1/jobs', json={'taskType': 'classification'})

    assert response.status_code == 400
    data = response.json()
    assert 'detail' not in data
    assert data['error']['code'] == 'VALIDATION_ERROR'
    assert data['error']['message'] == 'Request validation failed'
    assert 'requestId' in data['error']
    assert isinstance(data['error']['details']['fields'], list)


def test_extra_fields_are_rejected():
    """CreateJobRequest forbids unknown fields (e.g. the old exportFormats array)."""
    response = client.post(
        '/api/v1/jobs',
        json={
            'datasetId': '00000000-0000-0000-0000-000000000000',
            'featureColumns': ['a'],
            'labelColumn': 'label',
            'taskType': 'classification',
            'algorithmId': 'logistic_regression',
            'exportFormats': ['onnx', 'pkl'],
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data['error']['code'] == 'VALIDATION_ERROR'
    fields = data['error']['details']['fields']
    assert any('exportFormats' in str(error['loc']) for error in fields)


def test_model_validator_error_serialises():
    """A model_validator ValueError still yields a serialisable 400 envelope."""
    response = client.post(
        '/api/v1/jobs',
        json={
            'datasetId': '00000000-0000-0000-0000-000000000000',
            'featureColumns': ['a'],
            'taskType': 'classification',
            'algorithmId': 'logistic_regression',
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data['error']['code'] == 'VALIDATION_ERROR'
    assert isinstance(data['error']['details']['fields'], list)


def test_path_param_validation_uses_envelope():
    """Invalid path params (e.g. a non-UUID jobId) use the flat envelope."""
    response = client.get('/api/v1/jobs/not-a-uuid')

    assert response.status_code == 400
    data = response.json()
    assert 'detail' not in data
    assert data['error']['code'] == 'VALIDATION_ERROR'


def test_request_id_echoes_inbound_header():
    """The requestId in the envelope mirrors the inbound X-Request-Id header."""
    response = client.post(
        '/api/v1/jobs', json={'taskType': 'classification'}, headers={'X-Request-Id': 'req-123'}
    )

    assert response.status_code == 400
    assert response.json()['error']['requestId'] == 'req-123'
    assert response.headers['X-Request-Id'] == 'req-123'


def test_application_error_uses_flat_envelope():
    """Application errors (raise_http) also emit the flat envelope."""
    response = client.get('/api/v1/jobs', params={'limit': 0})

    assert response.status_code == 400
    data = response.json()
    assert 'detail' not in data
    assert data['error']['code'] == 'VALIDATION_ERROR'
    assert 'requestId' in data['error']


def test_unhandled_error_returns_internal_error_envelope():
    """An unexpected error returns 500 with the flat INTERNAL_ERROR envelope."""
    from fastapi import FastAPI
    from src.middleware.request_id import request_id_middleware

    probe = FastAPI()
    probe.middleware('http')(request_id_middleware)
    probe.add_exception_handler(Exception, app.exception_handlers[Exception])

    @probe.get('/boom')
    async def boom() -> None:
        raise RuntimeError('kaboom')

    with TestClient(probe, raise_server_exceptions=False) as probe_client:
        response = probe_client.get('/boom', headers={'X-Request-Id': 'err-500'})

    assert response.status_code == 500
    data = response.json()
    assert 'detail' not in data
    assert data['error']['code'] == 'INTERNAL_ERROR'
    assert data['error']['requestId'] == 'err-500'
