"""Unit tests for src/storage/filesystem.py."""

import hashlib
import uuid
from pathlib import Path

import pytest
from src.storage.filesystem import (
    artefact_dir,
    artefact_path,
    dataset_path,
    ensure_dirs,
    remove_job_files,
    sha256_of_file,
    storage_key_for,
    write_artefact,
    write_dataset,
)


@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch):
    """Override DATA_DIR for testing."""
    monkeypatch.setenv('SKL_DATA_DIR', str(tmp_path))
    # Force reload of the module to pick up new environment variable
    import importlib

    from src.storage import filesystem

    importlib.reload(filesystem)
    monkeypatch.setattr('src.storage.filesystem.DATA_DIR', tmp_path, raising=False)
    monkeypatch.setattr('src.storage.filesystem.DATASETS_DIR', tmp_path / 'datasets', raising=False)
    monkeypatch.setattr(
        'src.storage.filesystem.ARTEFACTS_DIR', tmp_path / 'artefacts', raising=False
    )
    return tmp_path


def test_ensure_dirs_creates_directories(temp_data_dir):
    """Test that ensure_dirs creates the required directories."""
    ensure_dirs()
    assert (temp_data_dir / 'datasets').exists()
    assert (temp_data_dir / 'artefacts').exists()


def test_storage_key_for():
    """Test storage key generation."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    key = storage_key_for('datasets', job_id, '.csv')
    assert key == (
        'datasets/12345678-1234-5678-1234-567812345678/12345678-1234-5678-1234-567812345678.csv'
    )


def test_sha256_of_file(tmp_path):
    """Test SHA-256 computation for a file."""
    test_file = tmp_path / 'test.txt'
    test_file.write_text('hello world')
    result = sha256_of_file(test_file)
    assert result == 'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'


def test_write_dataset(temp_data_dir, monkeypatch):
    """Test writing a dataset file."""
    # Monkeypatch the constants
    monkeypatch.setattr('src.storage.filesystem.DATASETS_DIR', temp_data_dir / 'datasets')
    monkeypatch.setattr('src.storage.filesystem.DATA_DIR', temp_data_dir)

    file_bytes = b'col1,col2\n1,2\n3,4\n'
    filename = 'test.csv'

    dataset_id, storage_key, sha256, byte_size = write_dataset(file_bytes, filename)

    assert isinstance(dataset_id, uuid.UUID)
    assert storage_key.startswith('datasets/')
    assert sha256 == hashlib.sha256(file_bytes).hexdigest()
    assert byte_size == len(file_bytes)

    # Verify file was written
    expected_path = (temp_data_dir / 'datasets') / storage_key
    assert expected_path.exists()
    assert expected_path.read_bytes() == file_bytes


def test_write_dataset_creates_dirs(temp_data_dir, monkeypatch):
    """Test that write_dataset creates parent directories."""
    monkeypatch.setattr('src.storage.filesystem.DATASETS_DIR', temp_data_dir / 'datasets')
    monkeypatch.setattr('src.storage.filesystem.DATA_DIR', temp_data_dir)

    file_bytes = b'col1,col2\n1,2\n'
    write_dataset(file_bytes, 'test.csv')
    assert (temp_data_dir / 'datasets').exists()


def test_dataset_path(temp_data_dir, monkeypatch):
    """Test dataset path resolution."""
    monkeypatch.setattr('src.storage.filesystem.DATASETS_DIR', temp_data_dir / 'datasets')
    storage_key = 'datasets/123/123.csv'
    result = dataset_path(storage_key)
    assert result == (temp_data_dir / 'datasets') / storage_key


def test_artefact_dir(temp_data_dir, monkeypatch):
    """Test artefact directory path."""
    monkeypatch.setattr('src.storage.filesystem.ARTEFACTS_DIR', temp_data_dir / 'artefacts')
    monkeypatch.setattr('src.storage.filesystem.DATA_DIR', temp_data_dir)

    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    result = artefact_dir(job_id)
    assert result == temp_data_dir / 'artefacts' / '12345678-1234-5678-1234-567812345678'


def test_write_artefact(temp_data_dir, monkeypatch):
    """Test writing an artefact file."""
    monkeypatch.setattr('src.storage.filesystem.ARTEFACTS_DIR', temp_data_dir / 'artefacts')
    monkeypatch.setattr('src.storage.filesystem.DATA_DIR', temp_data_dir)

    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    data = b'model data'
    filename = 'model.onnx'

    storage_key, path, byte_size = write_artefact(job_id, data, filename)

    assert storage_key.startswith('artefacts/')
    assert byte_size == len(data)
    assert Path(path).exists()
    assert Path(path).read_bytes() == data


def test_artefact_path(temp_data_dir, monkeypatch):
    """Test artefact path resolution."""
    monkeypatch.setattr('src.storage.filesystem.ARTEFACTS_DIR', temp_data_dir / 'artefacts')
    monkeypatch.setattr('src.storage.filesystem.DATA_DIR', temp_data_dir)

    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    storage_key = 'artefacts/123/123.onnx'
    result = artefact_path(storage_key, job_id)
    expected = (
        temp_data_dir
        / 'artefacts'
        / '12345678-1234-5678-1234-567812345678'
        / storage_key.replace('/', '_')
    )
    assert result == expected


def test_remove_job_files(temp_data_dir, monkeypatch):
    """Test removing job files."""
    monkeypatch.setattr('src.storage.filesystem.ARTEFACTS_DIR', temp_data_dir / 'artefacts')
    monkeypatch.setattr('src.storage.filesystem.DATA_DIR', temp_data_dir)

    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    adir = artefact_dir(job_id)
    adir.mkdir(parents=True, exist_ok=True)
    (adir / 'file1.txt').write_text('data1')
    (adir / 'file2.txt').write_text('data2')

    assert adir.exists()
    remove_job_files(job_id)
    assert not adir.exists()


def test_remove_job_files_nonexistent():
    """Test removing job files when directory doesn't exist (should not error)."""
    job_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    remove_job_files(job_id)  # Should not raise
