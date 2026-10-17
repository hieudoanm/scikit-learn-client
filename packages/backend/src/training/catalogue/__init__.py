"""Algorithm catalogue: task constants, minimums, and the estimator registry."""

from src.training.catalogue.registry import (
    _UNLABELLED_TOKENS,
    ALGORITHMS,
    LABEL_REQUIRED_TASKS,
    MIN_CLASSES,
    MIN_LABELLED_PER_CLASS,
    MIN_LABELLED_ROWS,
    RL_TASK_TYPE,
    TASK_TYPES,
)

__all__ = [
    'ALGORITHMS',
    'LABEL_REQUIRED_TASKS',
    'MIN_CLASSES',
    'MIN_LABELLED_PER_CLASS',
    'MIN_LABELLED_ROWS',
    'RL_TASK_TYPE',
    'TASK_TYPES',
    '_UNLABELLED_TOKENS',
]
