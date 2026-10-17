"""Unit tests for the artefact ZIP bundles (FEATURE-15, scenario 34)."""

import json
import uuid
import zipfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.api.jobs import download_jobs_bundle
from src.api.jobs_bundle import (
    SKIPPED_NO_ARTEFACTS,
    SKIPPED_NOT_COMPLETED,
    BundleJob,
    build_bundle,
    skipped_entry,
)
from src.api.jobs_lifecycle import download_job_bundle
from src.storage.filesystem import artefact_path


def _job(job_id=None, *, status='completed', name='Baseline'):
    """Build a mock job row."""
    job = MagicMock()
    job.id = job_id or uuid.uuid4()
    job.status = status
    job.name = name
    job.tags = ['baseline']
    job.task_type = 'classification'
    job.algorithm_id = 'logistic_regression'
    job.created_at = datetime(2024, 1, 1)
    job.library_versions = {'scikit-learn': '1.9.1'}
    return job


def _artefact(job_id, *, filename='model.onnx', fmt='onnx', storage_key=None):
    """Build a mock artefact row."""
    artefact = MagicMock()
    artefact.id = uuid.uuid4()
    artefact.job_id = job_id
    artefact.format = fmt
    artefact.filename = filename
    artefact.byte_size = 11
    artefact.storage_key = storage_key or f'artefacts/{job_id}/{filename}'
    return artefact


@pytest.fixture
def artefact_file(tmp_path, monkeypatch):
    """Write an artefact file on disk and point the bundle helper at it."""

    def _make(job_id, filename='model.onnx', content=b'model data'):
        path = tmp_path / f'{job_id}-{filename}'
        path.write_bytes(content)
        monkeypatch.setattr(
            'src.api.jobs_bundle.artefact_path',
            lambda key, owner: tmp_path / f'{owner}-{filename_of(key)}',
        )
        return path

    return _make


def filename_of(storage_key: str) -> str:
    """Return the file name part of a storage key."""
    return storage_key.rsplit('/', 1)[-1]


def test_build_bundle_writes_the_expected_entries(tmp_path, artefact_file):
    """The bundle holds the artefacts, the metrics, and a manifest per job."""
    job = _job()
    artefact = _artefact(job.id)
    artefact_file(job.id)

    bundle = build_bundle([BundleJob(job=job, artefacts=[artefact], metrics={'accuracy': 0.9})])
    try:
        with zipfile.ZipFile(bundle.path) as archive:
            names = archive.namelist()
            assert f'{job.id}/model.onnx' in names
            assert f'{job.id}/metrics.json' in names
            assert f'{job.id}/manifest.json' in names
            manifest = json.loads(archive.read(f'{job.id}/manifest.json'))
            assert manifest['jobId'] == str(job.id)
            assert manifest['algorithmId'] == 'logistic_regression'
            assert [entry['path'] for entry in manifest['files']] == ['model.onnx', 'metrics.json']
            assert json.loads(archive.read(f'{job.id}/metrics.json')) == {'accuracy': 0.9}
    finally:
        bundle.path.unlink(missing_ok=True)


def test_build_bundle_reports_a_missing_file(tmp_path, monkeypatch):
    """An artefact whose file is gone is reported, not silently dropped."""
    job = _job()
    artefact = _artefact(job.id)

    monkeypatch.setattr(
        'src.api.jobs_bundle.artefact_path',
        lambda key, job_id: tmp_path / 'does-not-exist.onnx',
    )

    bundle = build_bundle([BundleJob(job=job, artefacts=[artefact])])
    try:
        assert bundle.missing_files == [f'{job.id}/model.onnx']
        with zipfile.ZipFile(bundle.path) as archive:
            assert f'{job.id}/model.onnx' not in archive.namelist()
    finally:
        bundle.path.unlink(missing_ok=True)


def test_build_bundle_writes_the_skipped_list():
    """The skipped jobs are written to a top-level `skipped.json`."""
    job = _job()
    bundle = build_bundle(
        [BundleJob(job=job, artefacts=[])],
        skipped=[skipped_entry(job.id, SKIPPED_NO_ARTEFACTS)],
    )
    try:
        with zipfile.ZipFile(bundle.path) as archive:
            payload = json.loads(archive.read('skipped.json'))
            assert payload['skipped'][0]['jobId'] == str(job.id)
            assert payload['skipped'][0]['reason'] == SKIPPED_NO_ARTEFACTS
    finally:
        bundle.path.unlink(missing_ok=True)


def test_build_bundle_omits_the_skipped_file_when_nothing_was_skipped():
    """Without skipped jobs the archive carries no `skipped.json`."""
    job = _job()
    bundle = build_bundle([BundleJob(job=job, artefacts=[])])
    try:
        with zipfile.ZipFile(bundle.path) as archive:
            assert 'skipped.json' not in archive.namelist()
    finally:
        bundle.path.unlink(missing_ok=True)


def test_skipped_entry_shape():
    """A skipped entry names the job and the reason."""
    job_id = uuid.uuid4()
    entry = skipped_entry(job_id, SKIPPED_NOT_COMPLETED)
    assert entry == {'jobId': str(job_id), 'reason': SKIPPED_NOT_COMPLETED}


@pytest.mark.asyncio
async def test_download_job_bundle_returns_a_zip(mock_session, artefact_file):
    """The single-job bundle answers with a ZIP that carries the artefacts."""
    job = _job()
    artefact = _artefact(job.id)
    artefact_file(job.id)

    mock_job_result = MagicMock()
    mock_job_result.scalar_one_or_none = MagicMock(return_value=job)
    mock_artefacts_result = MagicMock()
    mock_artefacts_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[artefact]))
    )
    mock_metrics_result = MagicMock()
    mock_metrics_result.scalars = MagicMock(
        return_value=MagicMock(first=MagicMock(return_value=None))
    )
    mock_session.execute = AsyncMock(
        side_effect=[mock_job_result, mock_artefacts_result, mock_metrics_result]
    )

    response = await download_job_bundle(mock_session, job.id)

    assert response.media_type == 'application/zip'
    assert response.filename == f'{job.id}.zip'


@pytest.mark.asyncio
async def test_download_job_bundle_without_artefacts(mock_session):
    """A job without artefacts answers 404 ARTEFACT_NOT_FOUND."""
    job = _job()
    mock_job_result = MagicMock()
    mock_job_result.scalar_one_or_none = MagicMock(return_value=job)
    mock_artefacts_result = MagicMock()
    mock_artefacts_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[]))
    )
    mock_session.execute = AsyncMock(side_effect=[mock_job_result, mock_artefacts_result])

    with pytest.raises(Exception) as exc_info:
        await download_job_bundle(mock_session, job.id)
    assert 'ARTEFACT_NOT_FOUND' in str(exc_info.value)


@pytest.mark.asyncio
async def test_download_jobs_bundle_reports_an_unknown_job(mock_session):
    """An unknown job id answers 404 with the missing ids."""
    job = _job()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[job])))
    mock_session.execute = AsyncMock(return_value=mock_result)

    unknown = uuid.uuid4()
    with pytest.raises(Exception) as exc_info:
        await download_jobs_bundle(mock_session, f'{job.id},{unknown}')
    assert 'JOB_NOT_FOUND' in str(exc_info.value)
    assert str(unknown) in str(exc_info.value)


@pytest.mark.asyncio
async def test_download_jobs_bundle_without_any_artefact(mock_session):
    """When no selected job has artefacts the bundle answers 404."""
    job = _job(status='failed')
    mock_job_result = MagicMock()
    mock_job_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[job])))
    mock_artefacts_result = MagicMock()
    mock_artefacts_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[]))
    )
    mock_session.execute = AsyncMock(side_effect=[mock_job_result, mock_artefacts_result])

    with pytest.raises(Exception) as exc_info:
        await download_jobs_bundle(mock_session, str(job.id))
    assert 'ARTEFACT_NOT_FOUND' in str(exc_info.value)


def test_artefact_path_helper_is_importable():
    """The bundle resolves artefact files through the storage helper."""
    assert callable(artefact_path)
