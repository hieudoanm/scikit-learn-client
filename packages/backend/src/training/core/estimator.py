"""Estimator creation and export pipeline utilities."""

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import SelfTrainingClassifier

from src.training.catalogue import ALGORITHMS


def library_versions() -> dict[str, str]:
    """Return the pinned library versions used for training."""
    import importlib.metadata

    versions = {
        'scikit-learn': importlib.metadata.version('scikit-learn'),
        'numpy': importlib.metadata.version('numpy'),
        'pandas': importlib.metadata.version('pandas'),
        'joblib': importlib.metadata.version('joblib'),
        'python': (
            f'{__import__("sys").version_info.major}.'
            f'{__import__("sys").version_info.minor}.'
            f'{__import__("sys").version_info.micro}'
        ),
    }
    for pkg in ('skl2onnx', 'onnxruntime'):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def make_estimator(algorithm_id: str, hyperparameters: dict | None, seed: int = 42):
    """Clone a fresh estimator and apply validated hyperparameters and the job seed."""
    from src.training.core.validation import _hyperparameter_error

    estimator = clone(ALGORITHMS[algorithm_id]['estimator'])
    params = dict(hyperparameters or {})
    if algorithm_id == 'fitted_q_iteration':
        # n_iterations and gamma drive the FQI sweeps, not the Q regressor;
        # learning_rate and max_depth are applied as usual below.
        params.pop('n_iterations', None)
        params.pop('gamma', None)
    if 'random_state' in estimator.get_params(deep=False):
        params.setdefault('random_state', seed)
    if params:
        unknown = sorted(set(params) - set(estimator.get_params(deep=False)))
        if unknown:
            raise _hyperparameter_error(f'unknown keys: {unknown}')
        estimator.set_params(**params)
    return estimator


def build_export_pipeline(pipeline: Pipeline) -> Pipeline:
    """Return the pipeline that should be converted to ONNX.

    ``SelfTrainingClassifier`` has no skl2onnx converter, so the fitted base
    estimator is exported instead. Self-training changes only the training
    labels, so predictions match the wrapper.
    """
    estimator = pipeline.steps[-1][1]
    if isinstance(estimator, SelfTrainingClassifier):
        return Pipeline([*pipeline.steps[:-1], ('estimator', estimator.estimator_)])
    return pipeline
