"""Algorithm catalogue constants and the allow-listed estimator registry.

The estimator definitions live in :mod:`src.training.catalogue.algorithms`;
``ALGORITHMS`` is re-exported here so callers keep a single import site.
"""

from src.training.catalogue.algorithms import ALGORITHMS

__all__ = [
    'ALGORITHMS',
    'LABEL_REQUIRED_TASKS',
    'RL_TASK_TYPE',
    'TASK_TYPES',
]

TASK_TYPES = (
    'classification',
    'regression',
    'semi_supervised',
    'clustering',
    'anomaly_detection',
    'reinforcement_learning',
)
LABEL_REQUIRED_TASKS = frozenset({'classification', 'regression', 'semi_supervised'})
RL_TASK_TYPE = 'reinforcement_learning'

# Semi-supervised minimums (FR-082).
MIN_LABELLED_ROWS = 10
MIN_LABELLED_PER_CLASS = 2
MIN_CLASSES = 2

# Empty, whitespace-only, NaN, and null cells are unlabelled.
_UNLABELLED_TOKENS = frozenset({'', 'nan', 'null'})
