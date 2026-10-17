"""ZIP bundles of job artefacts (FEATURE-15).

The archive is built in a temporary file and then streamed, so the response
carries a ``Content-Length`` and nothing is ever written to the artefact volume.
"""

from __future__ import annotations

import json
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from src.db.models import Artefact, Job
from src.storage.filesystem import artefact_path

__all__ = [
    'SKIPPED_NOT_COMPLETED',
    'SKIPPED_NO_ARTEFACTS',
    'BundleJob',
    'BundleResult',
    'build_bundle',
]

SKIPPED_NO_ARTEFACTS = 'NO_ARTEFACTS'
SKIPPED_NOT_COMPLETED = 'NOT_COMPLETED'


@dataclass
class BundleJob:
    """One job and the artefacts/metrics that go into the bundle."""

    job: Job
    artefacts: list[Artefact] = field(default_factory=list)
    metrics: dict | None = None


@dataclass
class BundleResult:
    """The temporary archive plus the bookkeeping the endpoint reports."""

    path: Path
    missing_files: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


def _manifest(bundle_job: BundleJob, files: list[dict]) -> dict:
    """Build the ``manifest.json`` payload of one job."""
    job = bundle_job.job
    return {
        'jobId': str(job.id),
        'name': job.name,
        'tags': list(job.tags or []),
        'taskType': job.task_type,
        'algorithmId': job.algorithm_id,
        'status': job.status,
        'createdAt': job.created_at.isoformat(),
        'libraryVersions': job.library_versions or {},
        'files': files,
    }


def build_bundle(
    bundle_jobs: list[BundleJob],
    skipped: list[dict] | None = None,
) -> BundleResult:
    """Write a ZIP archive with one directory per job and return its path.

    ``skipped`` lists the jobs that were left out of the bundle; it is written
    to a top-level ``skipped.json`` when it is not empty.

    The caller owns the returned file and must delete it once the response has
    been sent.
    """
    missing: list[str] = []
    handle = tempfile.NamedTemporaryFile(prefix='skl-bundle-', suffix='.zip', delete=False)
    path = Path(handle.name)
    handle.close()

    try:
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for bundle_job in bundle_jobs:
                job = bundle_job.job
                folder = str(job.id)
                files: list[dict] = []
                for artefact in bundle_job.artefacts:
                    source = artefact_path(artefact.storage_key, job.id)
                    if not source.exists():
                        missing.append(f'{folder}/{artefact.filename}')
                        continue
                    archive.write(source, arcname=f'{folder}/{artefact.filename}')
                    files.append(
                        {
                            'path': artefact.filename,
                            'format': artefact.format,
                            'byteSize': artefact.byte_size,
                        }
                    )
                if bundle_job.metrics is not None:
                    payload = json.dumps(bundle_job.metrics, indent=2, sort_keys=True).encode()
                    archive.writestr(f'{folder}/metrics.json', payload)
                    files.append(
                        {'path': 'metrics.json', 'format': 'metrics', 'byteSize': len(payload)}
                    )
                manifest = json.dumps(_manifest(bundle_job, files), indent=2).encode()
                archive.writestr(f'{folder}/manifest.json', manifest)
            if skipped:
                payload = json.dumps({'skipped': skipped}, indent=2).encode()
                archive.writestr('skipped.json', payload)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    return BundleResult(path=path, missing_files=missing, skipped=list(skipped or []))


def skipped_entry(job_id: uuid.UUID, reason: str) -> dict:
    """Build one entry of the top-level ``skipped.json`` payload."""
    return {'jobId': str(job_id), 'reason': reason}
