"""Shared fixtures for the API endpoint tests."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def sample_dataset():
    """Create a mock dataset."""
    dataset = MagicMock()
    dataset.id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    dataset.columns = [
        {'name': 'feature1', 'inferredDtype': 'int64', 'sampleValues': [1, 2]},
        {'name': 'feature2', 'inferredDtype': 'int64', 'sampleValues': [3, 4]},
        {'name': 'label', 'inferredDtype': 'int64', 'sampleValues': [0, 1]},
    ]
    dataset.storage_key = 'datasets/123/test.csv'
    return dataset


@pytest.fixture
def sample_job():
    """Create a mock job."""
    job = MagicMock()
    job.id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    job.status = 'completed'
    job.dataset_id = uuid.UUID('87654321-8765-4321-8765-876543218765')
    job.task_type = 'classification'
    job.algorithm_id = 'logistic_regression'
    job.created_at = datetime(2024, 1, 1, 0, 0, 0)
    job.updated_at = datetime(2024, 1, 1, 1, 0, 0)
    job.feature_columns = ['feature1', 'feature2']
    job.label_column = 'label'
    job.rl_columns = None
    job.hyperparameters = {'C': 1.0}
    job.library_versions = {'scikit-learn': '1.9.1'}
    job.error_code = None
    job.error_message = None
    job.artefacts = []
    # Phase 12 metadata (FEATURE-17): name, tags, and the retry counters.
    job.name = 'Sample job'
    job.tags = ['baseline']
    job.attempts = 1
    job.max_attempts = 2
    job.next_attempt_at = None
    return job


@pytest.fixture
def sample_csv_bytes():
    """Sample CSV file bytes for testing."""
    return b'feature1,feature2,label\n1.0,2.0,0\n2.0,3.0,1\n3.0,1.0,0\n'


def empty_metrics_result():
    """A query result mock for a lookup that finds no metric row."""
    result = MagicMock()
    result.scalars = MagicMock(return_value=MagicMock(first=MagicMock(return_value=None)))
    return result


def empty_artefacts_result():
    """A query result mock for a lookup that finds no artefact row."""
    result = MagicMock()
    result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    return result
