"""Worker validation utilities."""

import pandas as pd

from src.db.models import Job
from src.training.catalogue import LABEL_REQUIRED_TASKS, RL_TASK_TYPE
from src.training.core.preprocessor import semi_supervised_problem, semi_supervised_stats


def _validate_mapping(df: pd.DataFrame, job: Job) -> None:
    """Validate the feature/label mapping against the dataset header."""
    header = set(df.columns.tolist())
    missing_x = [c for c in job.feature_columns if c not in header]
    if missing_x:
        raise ValueError(f'INVALID_COLUMN_MAP: feature columns not in dataset: {missing_x}')

    if job.task_type == RL_TASK_TYPE:
        if job.label_column:
            raise ValueError(
                'INVALID_COLUMN_MAP: reinforcement_learning jobs must not set a label column'
            )
    elif job.task_type in LABEL_REQUIRED_TASKS:
        if not job.label_column:
            raise ValueError(f'INVALID_COLUMN_MAP: a label column is required for {job.task_type}')
        if job.label_column not in header:
            raise ValueError(f'INVALID_COLUMN_MAP: label column not in dataset: {job.label_column}')
        if job.label_column in job.feature_columns:
            raise ValueError(
                f"INVALID_COLUMN_MAP: label column '{job.label_column}' is also a feature column"
            )
    elif job.label_column:
        raise ValueError(f'INVALID_COLUMN_MAP: {job.task_type} jobs must not set a label column')

    if job.task_type == 'regression' and job.label_column:
        if not pd.api.types.is_numeric_dtype(df[job.label_column]):
            raise ValueError(
                f"INVALID_LABEL_TYPE: label column '{job.label_column}' "
                'must be numeric for regression'
            )

    if job.task_type == 'semi_supervised' and job.label_column:
        problem = semi_supervised_problem(semi_supervised_stats(df[job.label_column]))
        if problem is not None:
            raise ValueError(f'INSUFFICIENT_LABELS: {problem}')
