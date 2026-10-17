"""Unit tests for src/api/health.py."""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_healthz_endpoint():
    """Test the healthz liveness endpoint."""
    response = client.get('/healthz')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_readyz_endpoint_with_db():
    """Test the readyz readiness endpoint (requires DB)."""
    # This test will fail if the database is not available
    # In a unit test context, we might want to mock the DB session
    response = client.get('/readyz')
    # Either 200 (DB available) or 503 (DB not available) is acceptable
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert response.json() == {'status': 'ok'}
    else:
        # Errors use the flat skl envelope, not FastAPI's nested 'detail'
        data = response.json()
        assert 'detail' not in data
        assert data['error']['code'] == 'SERVICE_UNAVAILABLE'
        assert 'requestId' in data['error']
