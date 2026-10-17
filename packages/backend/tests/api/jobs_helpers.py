"""Plain helper functions for the API endpoint tests (not fixtures)."""

from unittest.mock import AsyncMock, MagicMock


def single_dataset_result(session, dataset):
    """Point ``session.execute`` at a single dataset lookup result."""
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=dataset)
    session.execute = AsyncMock(return_value=result)
