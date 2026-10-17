"""Estimator creation and export pipeline utilities for skl CLI."""

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import SelfTrainingClassifier

from .catalogue.algorithms import ALGORITHMS


def make_estimator(algorithm_id: str, hyperparameters: dict | None, seed: int = 42):
    """Clone a fresh estimator and apply validated hyperparameters and the job seed."""
    estimator = clone(ALGORITHMS[algorithm_id]["estimator"])
    params = dict(hyperparameters or {})
    if algorithm_id == "fitted_q_iteration":
        params.pop("n_iterations", None)
        params.pop("gamma", None)
    if "random_state" in estimator.get_params(deep=False):
        params.setdefault("random_state", seed)
    if params:
        unknown = sorted(set(params) - set(estimator.get_params(deep=False)))
        if unknown:
            raise ValueError(f"INVALID_HYPERPARAMETERS: unknown keys: {unknown}")
        estimator.set_params(**params)
    return estimator


def build_export_pipeline(pipeline: Pipeline) -> Pipeline:
    """Return the pipeline that should be converted to ONNX."""
    estimator = pipeline.steps[-1][1]
    if isinstance(estimator, SelfTrainingClassifier):
        return Pipeline([*pipeline.steps[:-1], ("estimator", estimator.estimator)])
    return pipeline
