"""Database session management for skl."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import get_settings

from .models import Base


def _database_url() -> str:
    """Resolve the database URL from the central settings module."""
    return get_settings().database_url


engine = create_async_engine(_database_url(), echo=False, pool_pre_ping=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Yield an async database session."""
    async with async_session() as session:
        yield session


async def init_db() -> None:
    """Create all tables (for dev/testing; Alembic is source of truth in prod)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
