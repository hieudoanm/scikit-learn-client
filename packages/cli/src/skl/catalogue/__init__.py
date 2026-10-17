"""Algorithm catalogue for the skl CLI (mirrors the backend catalogue)."""

from .algorithms import ALGORITHMS

TASK_TYPES = (
    "classification",
    "regression",
    "semi_supervised",
    "clustering",
    "anomaly_detection",
    "reinforcement_learning",
)
LABEL_REQUIRED_TASKS = frozenset({"classification", "regression", "semi_supervised"})
RL_TASK_TYPE = "reinforcement_learning"


__all__ = [
    "ALGORITHMS",
    "LABEL_REQUIRED_TASKS",
    "RL_TASK_TYPE",
    "TASK_TYPES",
]
