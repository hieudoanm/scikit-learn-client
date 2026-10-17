"""Unit tests for the job comparison endpoint (FEATURE-14, scenario 33)."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.api.jobs import compare
from src.api.jobs_comparison import MAX_IDS, MIN_IDS, compare_jobs, parse_ids


def _job(
    job_id,
    *,
    dataset_id=None,
    task_type='classification',
    status='completed',
    feature_columns=('feature1',),
    error_code=None,
    error_message=None,
):
    """Build a mock job row."""
    job = MagicMock()
    job.id = job_id
    job.dataset_id = dataset_id or uuid.uuid4()
    job.task_type = task_type
    job.algorithm_id = 'logistic_regression'
    job.status = status
    job.name = 'Baseline'
    job.tags = ['baseline']
    job.feature_columns = list(feature_columns)
    job.label_column = 'label'
    job.split_ratio = 0.2
    job.random_seed = 42
    job.hyperparameters = {'C': 1.0}
    job.created_at = datetime(2024, 1, 1)
    job.error_code = error_code
    job.error_message = error_message
    return job


def _dataset(dataset_id):
    """Build a mock dataset row."""
    dataset = MagicMock()
    dataset.id = dataset_id
    dataset.original_filename = 'sample.csv'
    dataset.row_count = 100
    dataset.columns = [{'name': 'feature1'}, {'name': 'label'}]
    return dataset


def _result(rows):
    """Build a query result mock returning the given rows."""
    result = MagicMock()
    result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=rows)))
    return result


def test_parse_ids_requires_at_least_two():
    """Fewer than two ids is a validation error."""
    with pytest.raises(Exception) as exc_info:
        parse_ids(str(uuid.uuid4()))
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_parse_ids_rejects_more_than_five():
    """More than five ids is a validation error."""
    ids = ','.join(str(uuid.uuid4()) for _ in range(MAX_IDS + 1))
    with pytest.raises(Exception) as exc_info:
        parse_ids(ids)
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_parse_ids_rejects_a_malformed_id():
    """A malformed id is a validation error."""
    with pytest.raises(Exception) as exc_info:
        parse_ids(f'{uuid.uuid4()},not-a-uuid')
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_parse_ids_collapses_duplicates():
    """The same id twice counts once, so the minimum still applies."""
    job_id = uuid.uuid4()
    with pytest.raises(Exception) as exc_info:
        parse_ids(f'{job_id},{job_id}')
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_parse_ids_accepts_two_to_five():
    """Two to five distinct ids are accepted."""
    ids = [str(uuid.uuid4()) for _ in range(MIN_IDS)]
    assert len(parse_ids(','.join(ids))) == MIN_IDS


@pytest.mark.asyncio
async def test_compare_returns_one_entry_per_job():
    """The comparison carries one entry per requested job, in order."""
    first = _job(uuid.uuid4())
    second = _job(uuid.uuid4(), dataset_id=first.dataset_id)
    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[
            _result([first, second]),
            _result([_dataset(first.dataset_id)]),
            _result([]),
            _result([]),
        ]
    )

    response = await compare_jobs(session, [first.id, second.id])

    assert len(response.jobs) == 2
    assert response.jobs[0].jobId == first.id
    assert response.jobs[1].jobId == second.id
    assert response.warnings == []


@pytest.mark.asyncio
async def test_compare_reports_an_unknown_job():
    """An unknown job id answers 404 with the missing ids."""
    known = _job(uuid.uuid4())
    unknown = uuid.uuid4()
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[_result([known])])

    with pytest.raises(Exception) as exc_info:
        await compare_jobs(session, [known.id, unknown])
    assert 'JOB_NOT_FOUND' in str(exc_info.value)
    assert str(unknown) in str(exc_info.value)


@pytest.mark.asyncio
async def test_compare_warns_about_different_datasets():
    """Jobs from different datasets raise the DIFFERENT_DATASETS warning."""
    first = _job(uuid.uuid4())
    second = _job(uuid.uuid4())
    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[_result([first, second]), _result([]), _result([]), _result([])]
    )

    response = await compare_jobs(session, [first.id, second.id])

    assert 'DIFFERENT_DATASETS' in {warning.code for warning in response.warnings}


@pytest.mark.asyncio
async def test_compare_warns_about_different_tasks_and_features():
    """Different task types and feature sets raise their own warnings."""
    first = _job(uuid.uuid4(), task_type='classification', feature_columns=('a',))
    second = _job(uuid.uuid4(), task_type='regression', feature_columns=('b',))
    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[_result([first, second]), _result([]), _result([]), _result([])]
    )

    response = await compare_jobs(session, [first.id, second.id])

    codes = {warning.code for warning in response.warnings}
    assert 'DIFFERENT_TASKS' in codes
    assert 'DIFFERENT_FEATURES' in codes


@pytest.mark.asyncio
async def test_compare_reports_a_failed_job_without_metrics():
    """A failed job is compared with `metrics = null` and its error."""
    first = _job(uuid.uuid4(), status='failed', error_code='TRAINING_FAILED', error_message='boom')
    second = _job(uuid.uuid4(), dataset_id=first.dataset_id)
    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[
            _result([first, second]),
            _result([_dataset(first.dataset_id)]),
            _result([]),
            _result([]),
        ]
    )

    response = await compare_jobs(session, [first.id, second.id])

    failed = response.jobs[0]
    assert failed.metrics is None
    assert failed.error is not None
    assert failed.error.code == 'TRAINING_FAILED'
    assert 'INCOMPLETE_JOBS' in {warning.code for warning in response.warnings}


@pytest.mark.asyncio
async def test_compare_endpoint_parses_the_ids_parameter():
    """The endpoint parses the comma-separated `ids` parameter."""
    session = MagicMock()
    session.execute = AsyncMock(side_effect=AssertionError('must not be reached'))

    with pytest.raises(Exception) as exc_info:
        await compare(session, 'only-one')
    assert 'VALIDATION_ERROR' in str(exc_info.value)
