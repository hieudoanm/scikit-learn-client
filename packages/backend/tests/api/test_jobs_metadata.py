"""Unit tests for the job metadata editing (FEATURE-13, scenario 32)."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.api.jobs_lifecycle import update_job_metadata
from src.api.jobs_metadata import (
    MAX_NAME_LENGTH,
    MAX_TAG_LENGTH,
    MAX_TAGS,
    normalise_name,
    normalise_tags,
)
from src.api.schemas import UpdateJobMetadataRequest
from tests.api.conftest import empty_artefacts_result, empty_metrics_result


def test_normalise_name_trims():
    """A name is trimmed."""
    assert normalise_name('  Baseline run  ') == 'Baseline run'


def test_normalise_name_of_a_blank_string_is_none():
    """A blank name is stored as `null`."""
    assert normalise_name('   ') is None
    assert normalise_name('') is None
    assert normalise_name(None) is None


def test_normalise_name_rejects_an_overlong_name():
    """A name longer than the documented maximum is a validation error."""
    with pytest.raises(Exception) as exc_info:
        normalise_name('x' * (MAX_NAME_LENGTH + 1))
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'name' in str(exc_info.value)


def test_normalise_tags_lowercases_and_deduplicates():
    """Tags are lower-cased, trimmed, and de-duplicated, keeping the order."""
    assert normalise_tags(['Baseline', 'baseline', ' release ']) == ['baseline', 'release']


def test_normalise_tags_of_an_empty_list():
    """An empty list yields no tags."""
    assert normalise_tags([]) == []
    assert normalise_tags(None) == []


def test_normalise_tags_rejects_an_empty_tag():
    """An empty tag is a validation error."""
    with pytest.raises(Exception) as exc_info:
        normalise_tags(['baseline', '  '])
    assert 'VALIDATION_ERROR' in str(exc_info.value)
    assert 'tags' in str(exc_info.value)


def test_normalise_tags_rejects_an_overlong_tag():
    """A tag longer than the documented maximum is a validation error."""
    with pytest.raises(Exception) as exc_info:
        normalise_tags(['x' * (MAX_TAG_LENGTH + 1)])
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_normalise_tags_rejects_invalid_characters():
    """A tag may only hold letters, digits, spaces, and `-_.`."""
    with pytest.raises(Exception) as exc_info:
        normalise_tags(['baseline!'])
    assert 'VALIDATION_ERROR' in str(exc_info.value)


def test_normalise_tags_rejects_too_many_tags():
    """More than the documented maximum number of tags is a validation error."""
    with pytest.raises(Exception) as exc_info:
        normalise_tags([f'tag{index}' for index in range(MAX_TAGS + 1)])
    assert 'VALIDATION_ERROR' in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_job_metadata_stores_name_and_tags(mock_session, sample_job):
    """A PATCH stores the normalised name and tags."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=sample_job)
    mock_session.execute = AsyncMock(
        side_effect=[mock_result, empty_artefacts_result(), empty_metrics_result()]
    )

    body = UpdateJobMetadataRequest(name='  Baseline run  ', tags=['Baseline', 'release'])
    response = await update_job_metadata(mock_session, sample_job.id, body)

    assert sample_job.name == 'Baseline run'
    assert sample_job.tags == ['baseline', 'release']
    assert response.name == 'Baseline run'
    assert response.tags == ['baseline', 'release']


@pytest.mark.asyncio
async def test_update_job_metadata_leaves_absent_fields_unchanged(mock_session, sample_job):
    """A field that is absent from the request keeps its stored value."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=sample_job)
    mock_session.execute = AsyncMock(
        side_effect=[mock_result, empty_artefacts_result(), empty_metrics_result()]
    )

    await update_job_metadata(mock_session, sample_job.id, UpdateJobMetadataRequest(name='Renamed'))

    assert sample_job.name == 'Renamed'
    assert sample_job.tags == ['baseline']


@pytest.mark.asyncio
async def test_update_job_metadata_clears_the_name(mock_session, sample_job):
    """An empty name clears the stored name."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=sample_job)
    mock_session.execute = AsyncMock(
        side_effect=[mock_result, empty_artefacts_result(), empty_metrics_result()]
    )

    await update_job_metadata(mock_session, sample_job.id, UpdateJobMetadataRequest(name='  '))

    assert sample_job.name is None


@pytest.mark.asyncio
async def test_update_job_metadata_does_not_touch_the_training_configuration(
    mock_session, sample_job
):
    """The training configuration is immutable: the patch leaves it alone."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=sample_job)
    mock_session.execute = AsyncMock(
        side_effect=[mock_result, empty_artefacts_result(), empty_metrics_result()]
    )

    body = UpdateJobMetadataRequest(name='Renamed', tags=['baseline'])
    await update_job_metadata(mock_session, sample_job.id, body)

    assert sample_job.feature_columns == ['feature1', 'feature2']
    assert sample_job.label_column == 'label'
    assert sample_job.hyperparameters == {'C': 1.0}
    assert sample_job.algorithm_id == 'logistic_regression'


@pytest.mark.asyncio
async def test_update_job_metadata_of_an_unknown_job(mock_session):
    """Patching an unknown job answers 404 JOB_NOT_FOUND."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(Exception) as exc_info:
        await update_job_metadata(
            mock_session,
            uuid.UUID('12345678-1234-5678-1234-567812345678'),
            UpdateJobMetadataRequest(name='Renamed'),
        )
    assert 'JOB_NOT_FOUND' in str(exc_info.value)
