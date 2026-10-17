"""End-to-end tests for health, algorithms, and datasets (need a running server)."""

from tests.training.fixtures import CLASSIFICATION_IDS, REGRESSION_IDS


class TestHealth:
    async def test_healthz(self, http_client):
        response = await http_client.get('/healthz')
        assert response.status_code == 200
        assert response.json() == {'status': 'ok'}

    async def test_readyz(self, http_client):
        response = await http_client.get('/readyz')
        assert response.status_code == 200
        assert response.json() == {'status': 'ok'}


class TestAlgorithms:
    async def test_list_algorithms(self, http_client):
        response = await http_client.get('/api/v1/algorithms')
        assert response.status_code == 200
        data = response.json()
        assert 'algorithms' in data
        assert len(data['algorithms']) > 0
        algo_ids = {a['algorithmId'] for a in data['algorithms']}
        assert 'logistic_regression' in algo_ids
        assert 'random_forest_classifier' in algo_ids
        assert 'linear_regression' in algo_ids
        # Full catalogue, mirrored from the spec table
        # (see tests/training/catalogue/test_catalogue.py).
        assert CLASSIFICATION_IDS <= algo_ids
        assert REGRESSION_IDS <= algo_ids

    async def test_filter_by_task_type(self, http_client):
        response = await http_client.get('/api/v1/algorithms?taskType=classification')
        assert response.status_code == 200
        data = response.json()
        for algo in data['algorithms']:
            assert algo['taskType'] == 'classification'
        assert {a['algorithmId'] for a in data['algorithms']} == CLASSIFICATION_IDS


class TestDatasets:
    async def test_upload_dataset(self, http_client, dataset_csv):
        with open(dataset_csv, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            response = await http_client.post('/api/v1/datasets', files=files)
        assert response.status_code == 201
        data = response.json()
        assert data['filename'] == 'test_data.csv'
        assert data['byteSize'] > 0
        assert data['rowCount'] == 10
        assert len(data['columns']) == 3
        col_names = {c['name'] for c in data['columns']}
        assert col_names == {'feature1', 'feature2', 'label'}

    async def test_upload_empty_file(self, http_client):
        files = {'file': ('empty.csv', b'', 'text/csv')}
        response = await http_client.post('/api/v1/datasets', files=files)
        assert response.status_code == 400

    async def test_get_dataset(self, http_client, uploaded_dataset):
        response = await http_client.get(f'/api/v1/datasets/{uploaded_dataset}')
        assert response.status_code == 200
        data = response.json()
        assert data['datasetId'] == uploaded_dataset
