"""End-to-end job lifecycle tests (need a running server + Postgres)."""

import uuid


class TestJobs:
    async def test_create_job_validation_errors(self, http_client, uploaded_dataset):
        # Missing required fields
        response = await http_client.post('/api/v1/jobs', json={})
        assert response.status_code == 400

        # Unknown algorithm
        response = await http_client.post(
            '/api/v1/jobs',
            json={
                'datasetId': uploaded_dataset,
                'featureColumns': ['feature1'],
                'labelColumn': 'label',
                'taskType': 'classification',
                'algorithmId': 'unknown_algo',
            },
        )
        assert response.status_code == 400

        # Task-algorithm mismatch
        response = await http_client.post(
            '/api/v1/jobs',
            json={
                'datasetId': uploaded_dataset,
                'featureColumns': ['feature1'],
                'labelColumn': 'label',
                'taskType': 'regression',
                'algorithmId': 'logistic_regression',
            },
        )
        assert response.status_code == 400

    async def test_create_job_idempotency(self, http_client, uploaded_dataset):
        """Reusing an Idempotency-Key replays the 202; a changed body is 409."""
        payload = {
            'datasetId': uploaded_dataset,
            'featureColumns': ['feature1'],
            'labelColumn': 'label',
            'taskType': 'classification',
            'algorithmId': 'logistic_regression',
        }
        headers = {'Idempotency-Key': f'e2e-{uuid.uuid4().hex}'}

        first = await http_client.post('/api/v1/jobs', json=payload, headers=headers)
        assert first.status_code == 202
        first_id = first.json()['jobId']

        replay = await http_client.post('/api/v1/jobs', json=payload, headers=headers)
        assert replay.status_code == 202
        assert replay.json()['jobId'] == first_id

        conflict = await http_client.post(
            '/api/v1/jobs',
            json={**payload, 'algorithmId': 'random_forest_classifier'},
            headers=headers,
        )
        assert conflict.status_code == 409
        assert conflict.json()['error']['code'] == 'IDEMPOTENCY_KEY_EXISTS'

        await http_client.delete(f'/api/v1/jobs/{first_id}')

    async def test_full_training_flow(self, http_client, trained_job, db_pool):
        """Test the complete job lifecycle: create → train → complete → artefacts."""
        response = await http_client.get(f'/api/v1/jobs/{trained_job}')
        assert response.status_code == 200
        job = response.json()
        assert job['status'] == 'completed'
        assert job['metrics'] is not None
        assert 'accuracy' in job['metrics']
        assert len(job['artefacts']) == 3  # onnx + pkl + joblib

        formats = {a['format'] for a in job['artefacts']}
        assert formats == {'onnx', 'pkl', 'joblib'}

        row = await db_pool.fetchrow(
            'SELECT status FROM jobs WHERE id = $1', uuid.UUID(trained_job)
        )
        assert row['status'] == 'completed'

    async def test_download_artefact(self, http_client, trained_job):
        """Test downloading a completed job's artefact."""
        response = await http_client.get(f'/api/v1/jobs/{trained_job}')
        job = response.json()
        onnx_artefact = next(a for a in job['artefacts'] if a['format'] == 'onnx')

        response = await http_client.get(
            f'/api/v1/jobs/{trained_job}/artefacts/{onnx_artefact["artefactId"]}'
        )
        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/octet-stream'
        assert len(response.content) > 0

    async def test_list_jobs(self, http_client, trained_job):
        response = await http_client.get('/api/v1/jobs?limit=20')
        assert response.status_code == 200
        data = response.json()
        assert 'jobs' in data
        assert 'nextCursor' in data
        assert 'hasMore' in data
        assert len(data['jobs']) >= 1
        job_ids = {j['jobId'] for j in data['jobs']}
        assert trained_job in job_ids

    async def test_delete_job(self, http_client, uploaded_dataset):
        """Create a job, delete it, and verify it's gone."""
        payload = {
            'datasetId': uploaded_dataset,
            'featureColumns': ['feature1'],
            'labelColumn': 'label',
            'taskType': 'classification',
            'algorithmId': 'logistic_regression',
        }
        response = await http_client.post('/api/v1/jobs', json=payload)
        assert response.status_code == 202
        job_id = response.json()['jobId']

        response = await http_client.delete(f'/api/v1/jobs/{job_id}')
        assert response.status_code == 204

        response = await http_client.get(f'/api/v1/jobs/{job_id}')
        assert response.status_code == 404
