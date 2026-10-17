"""Pytest configuration for skl backend tests."""

import asyncio
import os
import re
import uuid

import asyncpg
import httpx
import pytest


def _database_url() -> str:
    """Test database URL: `DATABASE_URL` (set in CI) or the local default.

    asyncpg only speaks the plain `postgresql://` scheme, so SQLAlchemy-style
    driver suffixes such as `postgresql+asyncpg://` are stripped.
    """
    url = os.environ.get('DATABASE_URL', 'postgresql://skl:skl@localhost:5432/skl')
    url = re.sub(r'^(?:postgres|postgresql)\+[^:]+://', 'postgresql://', url)
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return url


DATABASE_URL = _database_url()


@pytest.fixture(scope='session')
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_pool():
    """Create a database connection pool."""
    pool = await asyncpg.create_pool(DATABASE_URL)
    yield pool
    await pool.close()


# --- API-level end-to-end fixtures (need a server on 127.0.0.1:8000) ----------

BASE_URL = 'http://127.0.0.1:8000'


@pytest.fixture
def dataset_csv(tmp_path):
    """Create a test CSV dataset file."""
    csv_path = tmp_path / 'test_data.csv'
    csv_path.write_text(
        'feature1,feature2,label\n'
        '1.0,2.0,0\n'
        '2.0,3.0,1\n'
        '3.0,1.0,0\n'
        '4.0,5.0,1\n'
        '5.0,4.0,0\n'
        '6.0,7.0,1\n'
        '7.0,6.0,0\n'
        '8.0,9.0,1\n'
        '9.0,8.0,0\n'
        '10.0,11.0,1\n'
    )
    return str(csv_path)


@pytest.fixture
async def http_client():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        yield client


@pytest.fixture
async def uploaded_dataset(http_client, dataset_csv):
    """Upload a test dataset and return its ID."""
    with open(dataset_csv, 'rb') as f:
        files = {'file': ('test_data.csv', f, 'text/csv')}
        response = await http_client.post('/api/v1/datasets', files=files)
    assert response.status_code == 201, f'Upload failed: {response.text}'
    data = response.json()
    assert 'datasetId' in data
    yield data['datasetId']

    # Cleanup: delete the dataset from the database
    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        await pool.execute('DELETE FROM datasets WHERE id = $1', uuid.UUID(data['datasetId']))
    finally:
        await pool.close()


@pytest.fixture
async def trained_job(http_client, uploaded_dataset):
    """Create and wait for a training job to complete."""
    payload = {
        'datasetId': uploaded_dataset,
        'featureColumns': ['feature1', 'feature2'],
        'labelColumn': 'label',
        'taskType': 'classification',
        'algorithmId': 'logistic_regression',
        'exportPkl': True,
        'exportJoblib': True,
    }
    response = await http_client.post('/api/v1/jobs', json=payload)
    assert response.status_code == 202, f'Job creation failed: {response.text}'
    job_data = response.json()
    job_id = job_data['jobId']

    # Poll until terminal status
    for _ in range(60):
        response = await http_client.get(f'/api/v1/jobs/{job_id}')
        assert response.status_code == 200
        job = response.json()
        status = job['status']
        if status in ('completed', 'failed'):
            break
        await asyncio.sleep(1)

    assert status == 'completed', (
        f'Job did not complete. Status: {status}, Error: {job.get("error")}'
    )
    yield job_id
