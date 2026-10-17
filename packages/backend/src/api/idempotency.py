"""Idempotent job creation (`Idempotency-Key` on POST /api/v1/jobs).

The first request carrying a key creates the job and caches its ``202`` body;
a repeat with the same key and an identical payload replays that body. Reusing
a key with a different payload is rejected with ``409 IDEMPOTENCY_KEY_EXISTS``.
Keys expire after :data:`KEY_TTL` and may then be reused.

Contract: docs/02-features/FEATURE-04-Training-Job.md and docs/06 §8.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import raise_http
from src.db.models import IdempotencyKey
from src.db.utils import utc_now

KEY_TTL = timedelta(hours=24)


def request_hash(body: dict[str, Any]) -> str:
    """Return the SHA-256 of the canonicalised request body.

    Keys are sorted and whitespace removed so that semantically identical
    payloads hash equally regardless of field order.
    """
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


async def lookup(session: AsyncSession, key: str, body_hash: str) -> dict | None:
    """Return the cached response body for ``key``, or ``None`` on a miss.

    Raises ``409 IDEMPOTENCY_KEY_EXISTS`` when the key is live but the payload
    hash differs. Expired rows are treated as absent.
    """
    result = await session.execute(select(IdempotencyKey).where(IdempotencyKey.key == key))
    record = result.scalar_one_or_none()
    if record is None:
        return None

    if record.expires_at <= utc_now():
        return None

    if record.request_hash != body_hash:
        raise_http(
            409,
            'IDEMPOTENCY_KEY_EXISTS',
            'Idempotency-Key already used with a different request body',
            details={'idempotencyKey': key},
        )

    return record.response_body


async def record(
    session: AsyncSession,
    key: str,
    body_hash: str,
    job_id: Any,
    response_body: dict,
) -> None:
    """Persist the ``202`` response under ``key`` for future replay."""
    await session.execute(delete(IdempotencyKey).where(IdempotencyKey.key == key))
    await session.execute(
        insert(IdempotencyKey).values(
            key=key,
            request_hash=body_hash,
            job_id=job_id,
            response_status=202,
            response_body=response_body,
            created_at=utc_now(),
            expires_at=utc_now() + KEY_TTL,
        )
    )


__all__ = ['KEY_TTL', 'lookup', 'record', 'request_hash']
