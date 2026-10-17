"""Feature importance extraction utilities."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def _get_feature_names(preprocessor, original_features: list[str]) -> list[str]:
    """Get expanded feature names from a fitted ColumnTransformer."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        # Fallback: use original feature names
        return original_features


def _aggregate_importance_to_original(
    importance: np.ndarray,
    expanded_names: list[str],
    original_features: list[str],
) -> dict[str, float]:
    """Aggregate expanded feature importances back to original feature columns.

    ColumnTransformer produces names like 'num__feature' or 'cat__feature_value'.
    We sum importances for each original feature.
    """
    aggregated: dict[str, float] = {f: 0.0 for f in original_features}

    for idx, name in enumerate(expanded_names):
        if idx >= len(importance):
            break
        # ColumnTransformer format: 'transformer__original_feature' or
        # 'transformer__original_feature_value'
        parts = name.split('__', 1)
        if len(parts) == 2:
            original = parts[1]
            # For one-hot encoded: 'cat__feature_value' -> original feature is 'feature'
            # The part after '__' may contain the category value
            # Find which original feature this belongs to
            for orig in original_features:
                if original == orig or original.startswith(orig + '_'):
                    aggregated[orig] += float(importance[idx])
                    break
            else:
                # Fallback: use the full name
                aggregated[original] = aggregated.get(original, 0.0) + float(importance[idx])
        else:
            # No transformer prefix
            aggregated[name] = aggregated.get(name, 0.0) + float(importance[idx])

    return aggregated


def _is_mlp(estimator) -> bool:
    """True for a fitted scikit-learn MLP (weights live in ``coefs_``)."""
    return isinstance(getattr(estimator, 'coefs_', None), list) and len(estimator.coefs_) > 0


def _mlp_first_layer_importance(estimator) -> np.ndarray:
    """Rank inputs by the MLP's first hidden layer weights.

    ``coefs_[0]`` has shape ``n_inputs x n_hidden``. The mean absolute outgoing
    weight per input ranks inputs the way ``coef_`` does for a linear model.
    """
    first_layer = np.asarray(estimator.coefs_[0])
    return np.mean(np.abs(first_layer), axis=1)


def _collect_feature_importance(
    pipeline: Pipeline,
    original_features: list[str],
    X_sample: pd.DataFrame,
    y_sample: np.ndarray | None = None,
    task_type: str = 'classification',
    seed: int = 42,
) -> dict[str, Any] | None:
    """Extract feature importance from the fitted pipeline.

    Supports:
    - Linear models (coef_)
    - Tree-based models (feature_importances_)
    - MLPs (first hidden layer of coefs_)
    - Permutation importance as fallback for other estimators
    """
    estimator = pipeline.named_steps['estimator']
    preprocessor = pipeline.named_steps['preprocessor']

    # Get expanded feature names
    expanded_names = _get_feature_names(preprocessor, original_features)

    importance: np.ndarray | None = None
    method = 'unknown'

    # 1. Try direct feature_importances_ (tree-based)
    if hasattr(estimator, 'feature_importances_'):
        importance = estimator.feature_importances_
        method = 'feature_importances_'
    # 2. Try coef_ (linear models) - use absolute value
    elif hasattr(estimator, 'coef_'):
        coef = estimator.coef_
        # For multi-class, average across classes
        if coef.ndim > 1:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
        method = 'coef_'
    # 2b. MLPs expose per-layer weight matrices in coefs_, not coef_.
    elif _is_mlp(estimator):
        importance = _mlp_first_layer_importance(estimator)
        method = 'mlp_weights'
    # 3. Try permutation importance as fallback (slow but model-agnostic)
    elif X_sample is not None and y_sample is not None and len(X_sample) > 0:
        try:
            # Use a small sample for speed
            n_permute = min(len(X_sample), 200)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_sample), n_permute, replace=False)
            X_perm = X_sample.iloc[idx]
            y_perm = y_sample[idx] if y_sample is not None else None

            result = permutation_importance(
                pipeline, X_perm, y_perm, n_repeats=5, random_state=seed, n_jobs=1
            )
            importance = result.importances_mean
            method = 'permutation_importance'
        except Exception:
            print(
                'Permutation importance calculation failed; using fallback importance.',
                exc_info=True,
            )
            pass

    if importance is None:
        return None

    # Aggregate to original features
    aggregated = _aggregate_importance_to_original(importance, expanded_names, original_features)

    # Sort by importance descending and take top 10
    sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        'feature': [f for f, _ in sorted_features],
        'importance': [float(v) for _, v in sorted_features],
        'method': method,
    }
