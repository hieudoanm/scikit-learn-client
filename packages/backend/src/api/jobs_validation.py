"""Job validation utilities."""

import uuid

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import select

from src.api.deps import raise_http
from src.api.schemas import CreateJobRequest
from src.db.models import Dataset
from src.storage.filesystem import dataset_path
from src.training.catalogue import ALGORITHMS, LABEL_REQUIRED_TASKS, TASK_TYPES
from src.training.core.preprocessor import semi_supervised_problem, semi_supervised_stats
from src.training.core.validation import validate_hyperparameters


async def validate_job_request(
    session,
    body: CreateJobRequest,
) -> tuple[uuid.UUID, Dataset, dict]:
    """Validate job creation request and return (dataset_id, dataset, validated_body)."""
    body_dict = body.model_dump()

    # --- required fields ---
    required = ['datasetId', 'featureColumns', 'taskType', 'algorithmId']
    missing = [
        k for k in required if k not in body_dict or body_dict[k] is None or body_dict[k] == ''
    ]
    if missing:
        raise_http(
            400,
            'VALIDATION_ERROR',
            f'Missing required fields: {missing}',
        )

    dataset_id_str = body_dict['datasetId']
    feature_columns = body_dict['featureColumns']
    label_column = body_dict.get('labelColumn')
    rl_columns = body_dict.get('rlColumns') or body_dict.get('rl_columns')
    task_type = body_dict['taskType']
    algorithm_id = body_dict['algorithmId']

    # --- validate dataset exists ---
    try:
        dataset_id = uuid.UUID(str(dataset_id_str))
    except ValueError, AttributeError:
        raise_http(400, 'VALIDATION_ERROR', 'Invalid datasetId')
    result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise_http(404, 'DATASET_NOT_FOUND', f'Dataset {dataset_id} not found')

    header = [c['name'] for c in dataset.columns]

    # --- validate algorithm ---
    if algorithm_id not in ALGORITHMS:
        raise_http(400, 'UNKNOWN_ALGORITHM', f'Unknown algorithm: {algorithm_id}')
    algo_spec = ALGORITHMS[algorithm_id]
    if algo_spec['task_type'] != task_type:
        raise_http(
            400,
            'TASK_ALGORITHM_MISMATCH',
            f'Algorithm {algorithm_id} is for {algo_spec["task_type"]}, not {task_type}',
        )

    # --- validate task type ---
    if task_type not in TASK_TYPES:
        raise_http(400, 'VALIDATION_ERROR', f'Invalid taskType: {task_type}')

    # --- validate label rules ---
    if task_type == 'reinforcement_learning':
        if label_column is not None and label_column != '':
            raise_http(
                400,
                'INVALID_COLUMN_MAP',
                'reinforcement_learning jobs must not set a label column',
                details={'labelColumn': label_column},
            )
        label_column = None
        if not rl_columns:
            raise_http(400, 'INVALID_COLUMN_MAP', 'rlColumns required for reinforcement_learning')
    else:
        if rl_columns is not None:
            raise_http(
                400, 'INVALID_COLUMN_MAP', 'rlColumns only allowed for reinforcement_learning'
            )
    if task_type in LABEL_REQUIRED_TASKS:
        if not label_column:
            raise_http(
                400,
                'INVALID_COLUMN_MAP',
                f'A label column is required for {task_type}',
                details={'taskType': task_type},
            )
    elif task_type not in LABEL_REQUIRED_TASKS and label_column:
        raise_http(
            400,
            'INVALID_COLUMN_MAP',
            f'{task_type} jobs must not set a label column',
            details={'labelColumn': label_column},
        )

    # --- validate mapping ---
    missing_x = [c for c in feature_columns if c not in header]
    if missing_x:
        raise_http(
            400,
            'INVALID_COLUMN_MAP',
            f'Feature columns not in dataset: {missing_x}',
            details={'missing': missing_x},
        )
    if label_column and label_column not in header:
        raise_http(
            400,
            'INVALID_COLUMN_MAP',
            f'Label column not in dataset: {label_column}',
            details={'labelColumn': label_column},
        )
    if label_column and label_column in feature_columns:
        raise_http(
            400,
            'INVALID_COLUMN_MAP',
            f"Label column '{label_column}' is also a feature column",
        )

    if task_type == 'regression':
        # Light type check: try to confirm label is numeric via a quick preview read
        try:
            path = dataset_path(dataset.storage_key)
            df_sample = pd.read_csv(path, nrows=10)
            if label_column in df_sample.columns and not pd.api.types.is_numeric_dtype(
                df_sample[label_column]
            ):
                raise_http(
                    400,
                    'INVALID_LABEL_TYPE',
                    f"Label column '{label_column}' must be numeric for regression",
                )
        except HTTPException:
            raise
        except Exception:  # noqa: S110
            pass  # Don't block job creation on preview read issues

    hyperparameters = body_dict.get('hyperparameters', {}) or {}
    row_count = dataset.row_count if isinstance(dataset.row_count, int) else None
    try:
        validate_hyperparameters(algorithm_id, hyperparameters, row_count=row_count)
    except ValueError as exc:
        raise_http(
            400,
            'INVALID_HYPERPARAMETERS',
            str(exc).removeprefix('INVALID_HYPERPARAMETERS: '),
        )

    if task_type == 'semi_supervised':
        try:
            labels = pd.read_csv(dataset_path(dataset.storage_key), usecols=[label_column])[
                label_column
            ]
        except Exception as exc:  # noqa: BLE001
            raise_http(400, 'VALIDATION_ERROR', f'Unable to read label column: {exc}')
        stats = semi_supervised_stats(labels)
        problem = semi_supervised_problem(stats)
        if problem is not None:
            raise_http(400, 'INSUFFICIENT_LABELS', problem, details=stats)

    # Return validated values
    validated = {
        'feature_columns': feature_columns,
        'label_column': label_column,
        'rl_columns': rl_columns if rl_columns is not None else None,
        'hyperparameters': hyperparameters,
        'split_ratio': body_dict.get('splitRatio', 0.2),
        'random_seed': body_dict.get('randomSeed', 42),
        'export_pkl': body_dict.get('exportPkl', False),
        'export_joblib': body_dict.get('exportJoblib', False),
    }

    return dataset_id, dataset, validated
