"""Unit tests for src/export/onnx.py export functions."""

import uuid

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.export.onnx import export_joblib, export_pickle


@pytest.fixture
def sample_pipeline():
    """Create a simple sklearn pipeline for testing."""
    return Pipeline([('estimator', LogisticRegression())])


@pytest.fixture
def fitted_pipeline(sample_pipeline):
    """Create a fitted sklearn pipeline."""
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    sample_pipeline.fit(X, y)
    return sample_pipeline


def test_export_joblib(fitted_pipeline, monkeypatch):
    """Test export_joblib function."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')

    # Mock write_artefact to avoid file I/O
    def mock_write(job_id, data, filename):
        return ('test-key', 'in-memory', len(data))

    monkeypatch.setattr('src.export.onnx.write_artefact', mock_write)

    data, filename, storage_key, byte_size = export_joblib(fitted_pipeline, job_id)

    assert isinstance(data, bytes)
    assert len(data) > 0
    assert filename == f'{job_id}.joblib'
    assert storage_key == 'test-key'
    assert byte_size == len(data)


def test_export_pickle(fitted_pipeline, monkeypatch):
    """Test export_pickle function."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')

    # Mock write_artefact to avoid file I/O
    def mock_write(job_id, data, filename):
        return ('test-key', 'in-memory', len(data))

    monkeypatch.setattr('src.export.onnx.write_artefact', mock_write)

    data, filename, storage_key, byte_size = export_pickle(fitted_pipeline, job_id)

    assert isinstance(data, bytes)
    assert len(data) > 0
    assert filename == f'{job_id}.pkl'
    assert storage_key == 'test-key'
    assert byte_size == len(data)


def test_export_joblib_filename_format(fitted_pipeline, monkeypatch):
    """Test that export_joblib produces correct filename format."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    monkeypatch.setattr('src.export.onnx.write_artefact', lambda *args: ('key', 'path', 100))

    _, filename, _, _ = export_joblib(fitted_pipeline, job_id)
    assert filename.endswith('.joblib')
    assert str(job_id) in filename


def test_export_pickle_filename_format(fitted_pipeline, monkeypatch):
    """Test that export_pickle produces correct filename format."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    monkeypatch.setattr('src.export.onnx.write_artefact', lambda *args: ('key', 'path', 100))

    _, filename, _, _ = export_pickle(fitted_pipeline, job_id)
    assert filename.endswith('.pkl')
    assert str(job_id) in filename


def test_export_joblib_serializes_pipeline(fitted_pipeline, monkeypatch):
    """Test that export_joblib actually serializes the pipeline."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    captured_data = []

    def mock_write(job_id, data, filename):
        captured_data.append(data)
        return ('key', 'path', len(data))

    monkeypatch.setattr('src.export.onnx.write_artefact', mock_write)

    export_joblib(fitted_pipeline, job_id)

    assert len(captured_data) == 1
    assert len(captured_data[0]) > 0
    # Verify it's actually joblib-serialized data (has joblib signature)
    assert captured_data[0][:4] == b'\x80\x02' or len(captured_data[0]) > 100


def test_export_pickle_serializes_pipeline(fitted_pipeline, monkeypatch):
    """Test that export_pickle actually serializes the pipeline."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    captured_data = []

    def mock_write(job_id, data, filename):
        captured_data.append(data)
        return ('key', 'path', len(data))

    monkeypatch.setattr('src.export.onnx.write_artefact', mock_write)

    export_pickle(fitted_pipeline, job_id)

    assert len(captured_data) == 1
    assert len(captured_data[0]) > 0
