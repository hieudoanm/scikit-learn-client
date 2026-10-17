"""Unit tests for src/api/algorithms.py."""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_list_algorithms_no_filter():
    """Test listing all algorithms without task type filter."""
    response = client.get('/api/v1/algorithms')
    assert response.status_code == 200
    data = response.json()
    assert 'algorithms' in data
    assert len(data['algorithms']) > 0


def test_list_algorithms_filter_classification():
    """Test listing algorithms filtered by classification task type."""
    response = client.get('/api/v1/algorithms?taskType=classification')
    assert response.status_code == 200
    data = response.json()
    assert 'algorithms' in data
    for algo in data['algorithms']:
        assert algo['taskType'] == 'classification'


def test_list_algorithms_filter_regression():
    """Test listing algorithms filtered by regression task type."""
    response = client.get('/api/v1/algorithms?taskType=regression')
    assert response.status_code == 200
    data = response.json()
    assert 'algorithms' in data
    for algo in data['algorithms']:
        assert algo['taskType'] == 'regression'


def test_list_algorithms_response_structure():
    """Test that algorithm entries have the required structure."""
    response = client.get('/api/v1/algorithms')
    assert response.status_code == 200
    data = response.json()
    assert 'algorithms' in data
    if data['algorithms']:
        algo = data['algorithms'][0]
        assert 'algorithmId' in algo
        assert 'name' in algo
        assert 'taskType' in algo
        assert 'family' in algo
        assert 'description' in algo
        assert 'defaultHyperparameters' in algo
        assert 'exports' in algo
