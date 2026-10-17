"""Hyperparameter validation utilities."""

from src.training.catalogue import ALGORITHMS


def _hyperparameter_error(message: str) -> ValueError:
    return ValueError(f'INVALID_HYPERPARAMETERS: {message}')


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_cluster_count(key: str, value: object, row_count: int | None) -> None:
    if not _is_int(value) or value < 2:
        raise _hyperparameter_error(f'{key} must be an integer >= 2')
    if row_count is not None and value > row_count - 1:
        raise _hyperparameter_error(f'{key} must be <= rows - 1 ({row_count - 1})')


_MLP_ACTIVATIONS = {'identity', 'logistic', 'tanh', 'relu'}
_MLP_SOLVERS = {'lbfgs', 'sgd', 'adam'}


def _validate_mlp_hyperparameters(value) -> None:
    """Validate MLP architecture ranges (FR-023, FR-027)."""
    sizes = value('hidden_layer_sizes', [100])
    if not isinstance(sizes, list) or not 1 <= len(sizes) <= 5:
        raise _hyperparameter_error('hidden_layer_sizes must be a list of 1 to 5 integers')
    if not all(_is_int(n) and n >= 1 for n in sizes):
        raise _hyperparameter_error('hidden_layer_sizes must be integers >= 1')

    activation = value('activation', 'relu')
    if activation not in _MLP_ACTIVATIONS:
        raise _hyperparameter_error(f'activation must be one of {sorted(_MLP_ACTIVATIONS)}')

    solver = value('solver', 'adam')
    if solver not in _MLP_SOLVERS:
        raise _hyperparameter_error(f'solver must be one of {sorted(_MLP_SOLVERS)}')

    alpha = value('alpha', 0.0001)
    if not _is_number(alpha) or float(alpha) < 0:
        raise _hyperparameter_error('alpha must be a number >= 0')

    learning_rate_init = value('learning_rate_init', 0.001)
    if not _is_number(learning_rate_init) or not (0 < float(learning_rate_init) <= 1):
        raise _hyperparameter_error('learning_rate_init must be in (0, 1]')

    max_iter = value('max_iter', 300)
    if not _is_int(max_iter) or max_iter < 1:
        raise _hyperparameter_error('max_iter must be an integer >= 1')

    batch_size = value('batch_size', 'auto')
    if batch_size != 'auto' and (not _is_int(batch_size) or batch_size < 1):
        raise _hyperparameter_error('batch_size must be "auto" or an integer >= 1')

    validation_fraction = value('validation_fraction', 0.1)
    if not _is_number(validation_fraction) or not (0 < float(validation_fraction) < 1):
        raise _hyperparameter_error('validation_fraction must be in (0, 1)')

    early_stopping = value('early_stopping', False)
    if not isinstance(early_stopping, bool):
        raise _hyperparameter_error('early_stopping must be a boolean')


def validate_hyperparameters(
    algorithm_id: str,
    hyperparameters: dict,
    row_count: int | None = None,
) -> None:
    """Validate hyperparameter keys and task-specific ranges (FR-027)."""
    if not isinstance(hyperparameters, dict):
        raise _hyperparameter_error('hyperparameters must be an object')

    estimator = ALGORITHMS[algorithm_id]['estimator']
    allowed = set(estimator.get_params(deep=False))
    if algorithm_id == 'fitted_q_iteration':
        # FQI sweeps are consumed by the RL trainer, not the Q regressor.
        allowed |= {'n_iterations', 'gamma', 'max_depth', 'learning_rate'}
    unknown = sorted(set(hyperparameters) - allowed)
    if unknown:
        raise _hyperparameter_error(f'unknown keys: {unknown}')

    def value(key: str, default):
        return hyperparameters.get(key, default)

    if algorithm_id == 'self_training_classifier':
        threshold = value('threshold', 0.75)
        if not _is_number(threshold) or not 0.5 <= float(threshold) <= 1.0:
            raise _hyperparameter_error('threshold must be between 0.5 and 1.0')
        max_iter = value('max_iter', 10)
        if not _is_int(max_iter) or max_iter < 1:
            raise _hyperparameter_error('max_iter must be an integer >= 1')

    if algorithm_id in {'kmeans', 'mini_batch_kmeans'}:
        k = value('n_clusters', 3)
        _validate_cluster_count('n_clusters', k, row_count)

    if algorithm_id == 'gaussian_mixture':
        k = value('n_components', 3)
        _validate_cluster_count('n_components', k, row_count)

    if algorithm_id == 'isolation_forest':
        contamination = value('contamination', 'auto')
        if contamination != 'auto':
            if not _is_number(contamination) or not 0 < float(contamination) <= 0.5:
                raise _hyperparameter_error('contamination must be "auto" or in (0, 0.5]')
        n_estimators = value('n_estimators', 100)
        if not _is_int(n_estimators) or n_estimators < 1:
            raise _hyperparameter_error('n_estimators must be an integer >= 1')

    if algorithm_id in {'mlp_classifier', 'mlp_regressor'}:
        _validate_mlp_hyperparameters(value)

    if algorithm_id == 'fitted_q_iteration':
        n_iterations = value('n_iterations', 10)
        if not _is_int(n_iterations) or not 1 <= int(n_iterations) <= 50:
            raise _hyperparameter_error('n_iterations must be an integer between 1 and 50')
        gamma = value('gamma', 0.9)
        if not _is_number(gamma) or not 0 <= float(gamma) <= 1:
            raise _hyperparameter_error('gamma must be between 0 and 1')
        max_depth = value('max_depth', 3)
        if not _is_int(max_depth) or not 1 <= int(max_depth) <= 20:
            raise _hyperparameter_error('max_depth must be an integer between 1 and 20')
        learning_rate = value('learning_rate', 0.1)
        if not _is_number(learning_rate) or not (0 < float(learning_rate) <= 1):
            raise _hyperparameter_error('learning_rate must be greater than 0 and <= 1')
