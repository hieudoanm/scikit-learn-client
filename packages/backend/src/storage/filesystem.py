"""Filesystem storage operations for skl artefacts and datasets."""

import hashlib
import shutil
import uuid
from pathlib import Path

from src.config import get_settings

_settings = get_settings()

DATA_DIR: Path = _settings.data_dir
DATASETS_DIR: Path = DATA_DIR / 'datasets'
ARTEFACTS_DIR: Path = DATA_DIR / 'artefacts'
MAX_UPLOAD_BYTES: int = _settings.max_upload_bytes


def ensure_dirs() -> None:
    """Ensure the storage directories exist."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)


def storage_key_for(prefix: str, entity_id: uuid.UUID, suffix: str) -> str:
    """Build a deterministic, safe storage key."""
    return f'{prefix}/{entity_id}/{entity_id}{suffix}'


def sha256_of_file(path: Path | str) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_dataset(file_bytes: bytes, filename: str) -> tuple[uuid.UUID, str, str, int]:
    """Write an uploaded dataset to the volume.

    Returns (dataset_id, storage_key, sha256, byte_size).
    """
    ensure_dirs()
    dataset_id = uuid.uuid4()
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    storage_key = storage_key_for('datasets', dataset_id, '.csv')
    dest = DATASETS_DIR / storage_key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file_bytes)
    return dataset_id, str(storage_key), sha256, len(file_bytes)


def dataset_path(storage_key: str) -> Path:
    """Return the filesystem path for a dataset storage key."""
    return DATASETS_DIR / storage_key


def artefact_dir(job_id: uuid.UUID) -> Path:
    """Return the artefacts directory for a job."""
    return ARTEFACTS_DIR / str(job_id)


def write_artefact(job_id: uuid.UUID, data: bytes, filename: str) -> tuple[str, str, int]:
    """Write an artefact file and return (storage_key, path, byte_size)."""
    ensure_dirs()
    adir = artefact_dir(job_id)
    adir.mkdir(parents=True, exist_ok=True)
    storage_key = storage_key_for('artefacts', job_id, f'.{filename.split(".")[-1]}')
    dest = adir / storage_key.replace('/', '_')
    dest.write_bytes(data)
    return str(storage_key), str(dest), len(data)


def artefact_path(storage_key: str, job_id: uuid.UUID) -> Path:
    """Return the filesystem path for an artefact storage key."""
    return artefact_dir(job_id) / storage_key.replace('/', '_')


def remove_job_files(job_id: uuid.UUID) -> None:
    """Remove all files associated with a job from the volume."""
    adir = artefact_dir(job_id)
    if adir.exists():
        shutil.rmtree(adir, ignore_errors=True)
