"""Unit tests for src/db/session.py."""

import pytest
from src.config import reset_settings
from src.db.session import _database_url, async_session, engine, get_session


@pytest.fixture(autouse=True)
def _reset_settings():
    """Drop the cached settings so every test resolves the environment afresh."""
    reset_settings()
    yield
    reset_settings()


def test_database_url_from_env(monkeypatch):
    """Test _database_url with DATABASE_URL set."""
    monkeypatch.setenv('DATABASE_URL', 'postgres://user:pass@host:5432/db')
    result = _database_url()
    assert result == 'postgresql+asyncpg://user:pass@host:5432/db'


def test_database_url_postgresql_scheme(monkeypatch):
    """Test _database_url with postgresql:// scheme."""
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pass@host:5432/db')
    result = _database_url()
    assert result == 'postgresql+asyncpg://user:pass@host:5432/db'


def test_database_url_postgresql_with_asyncpg(monkeypatch):
    """Test _database_url with postgresql+asyncpg:// scheme (no change)."""
    monkeypatch.setenv('DATABASE_URL', 'postgresql+asyncpg://user:pass@host:5432/db')
    result = _database_url()
    assert result == 'postgresql+asyncpg://user:pass@host:5432/db'


def test_database_url_default(monkeypatch):
    """Test _database_url with no environment variable (default)."""
    monkeypatch.delenv('DATABASE_URL', raising=False)
    result = _database_url()
    assert result == 'postgresql+asyncpg://skl:skl@localhost:5432/skl'


def test_engine_exists():
    """Test that engine is created."""
    assert engine is not None


def test_async_session_exists():
    """Test that async_session is created."""
    assert async_session is not None


def test_get_session_is_generator():
    """Test that get_session returns an async generator."""
    import inspect

    assert inspect.isasyncgenfunction(get_session)


@pytest.mark.asyncio
async def test_get_session_yields_session():
    """Test that get_session yields a session."""
    from sqlalchemy.ext.asyncio import AsyncSession

    gen = get_session()
    session = await gen.__anext__()
    assert isinstance(session, AsyncSession)
    await gen.aclose()
