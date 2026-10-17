"""Unit tests for semi-supervised and RL column validation (FR-081/082/111)."""

import uuid
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from src.api.jobs_lifecycle import create_job
from src.api.schemas import CreateJobRequest, JobAcceptedResponse, RLColumns, TaskType
from tests.api.jobs_helpers import single_dataset_result


@pytest.mark.asyncio
async def test_create_job_semi_supervised_success(
    mock_session, sample_dataset, monkeypatch, tmp_path
):
    """Semi-supervised jobs accept empty cells as unlabelled rows (FR-081)."""
    csv_path = tmp_path / 'semi.csv'
    rows = ['feature1,feature2,label']
    for index in range(20):
        label = '' if index % 4 == 0 else ('yes' if index % 2 == 0 else 'no')
        rows.append(f'{index},{index + 1},{label}')
    csv_path.write_text('\n'.join(rows) + '\n')

    sample_dataset.row_count = 20
    single_dataset_result(mock_session, sample_dataset)
    monkeypatch.setattr('src.api.jobs_validation.dataset_path', lambda key: csv_path)
    monkeypatch.setattr('src.api.jobs_lifecycle.enqueue_job', MagicMock())

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1', 'feature2'],
        labelColumn='label',
        taskType=TaskType.SEMI_SUPERVISED,
        algorithmId='self_training_classifier',
    )

    result = await create_job(mock_session, body)

    assert isinstance(result, JobAcceptedResponse)
    assert result.taskType == TaskType.SEMI_SUPERVISED
    assert result.algorithmId == 'self_training_classifier'


@pytest.mark.asyncio
async def test_create_job_semi_supervised_insufficient_labels(
    mock_session, sample_dataset, monkeypatch, tmp_path
):
    """Fewer than 10 labelled rows returns INSUFFICIENT_LABELS (FR-082)."""
    csv_path = tmp_path / 'semi.csv'
    rows = ['feature1,feature2,label']
    for index in range(20):
        label = 'yes' if index < 4 else ''
        rows.append(f'{index},{index + 1},{label}')
    csv_path.write_text('\n'.join(rows) + '\n')

    sample_dataset.row_count = 20
    single_dataset_result(mock_session, sample_dataset)
    monkeypatch.setattr('src.api.jobs_validation.dataset_path', lambda key: csv_path)

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1', 'feature2'],
        labelColumn='label',
        taskType=TaskType.SEMI_SUPERVISED,
        algorithmId='self_training_classifier',
    )

    with pytest.raises(Exception) as exc_info:
        await create_job(mock_session, body)
    assert 'INSUFFICIENT_LABELS' in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_job_semi_supervised_unreadable_label(
    mock_session, sample_dataset, monkeypatch
):
    """A label column that cannot be read returns VALIDATION_ERROR."""
    single_dataset_result(mock_session, sample_dataset)
    monkeypatch.setattr(
        'src.api.jobs_validation.dataset_path',
        MagicMock(side_effect=FileNotFoundError('missing')),
    )

    body = CreateJobRequest(
        datasetId=sample_dataset.id,
        featureColumns=['feature1', 'feature2'],
        labelColumn='label',
        taskType=TaskType.SEMI_SUPERVISED,
        algorithmId='self_training_classifier',
    )

    with pytest.raises(Exception) as exc_info:
        await create_job(mock_session, body)
    assert 'VALIDATION_ERROR' in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_job_rl_requires_rl_columns():
    """rlColumns is required for reinforcement_learning (FR-111) - validated by Pydantic."""
    with pytest.raises(ValidationError) as exc_info:
        CreateJobRequest(
            datasetId=uuid.uuid4(),
            featureColumns=['feature1', 'feature2'],
            taskType=TaskType.REINFORCEMENT_LEARNING,
            algorithmId='fitted_q_iteration',
        )
    assert 'rlColumns required' in str(exc_info.value) or (
        'RLColumns required' in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_create_job_rl_rejects_label_column():
    """RL jobs must not set labelColumn (FR-111) - validated by Pydantic."""
    with pytest.raises(ValidationError) as exc_info:
        CreateJobRequest(
            datasetId=uuid.uuid4(),
            featureColumns=['feature1', 'feature2'],
            labelColumn='label',
            taskType=TaskType.REINFORCEMENT_LEARNING,
            algorithmId='fitted_q_iteration',
            rlColumns=RLColumns(actionColumn='a', rewardColumn='r'),
        )
    assert 'must not set labelColumn' in str(exc_info.value) or (
        'reinforcement_learning must not set labelColumn' in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_create_job_rl_columns_rejected_for_other_tasks():
    """rlColumns is rejected for non-RL tasks (FR-111) - validated by Pydantic."""
    with pytest.raises(ValidationError) as exc_info:
        CreateJobRequest(
            datasetId=uuid.uuid4(),
            featureColumns=['feature1', 'feature2'],
            labelColumn='label',
            taskType=TaskType.CLASSIFICATION,
            algorithmId='logistic_regression',
            rlColumns=RLColumns(actionColumn='a', rewardColumn='r'),
        )
    assert 'rlColumns only allowed' in str(exc_info.value) or (
        'only allowed for reinforcement' in str(exc_info.value)
    )
