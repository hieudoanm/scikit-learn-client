"""Shared helpers for the training test suites.

Kept import-light (no test bodies) so the catalogue, core, and RL test modules
can share the id sets, dataset/job builders, and ONNX helpers.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pandas as pd
from src.db.models import Job

CLASSIFICATION_IDS = {
    'logistic_regression',
    'random_forest_classifier',
    'gradient_boosting_classifier',
    'decision_tree_classifier',
    'extra_trees_classifier',
    'k_neighbors_classifier',
    'gaussian_naive_bayes',
    'svm_classifier',
    'mlp_classifier',
    'ada_boost_classifier',
    'bagging_classifier',
    'bernoulli_nb',
    'calibrated_classifier_cv',
    'categorical_nb',
    'complement_nb',
    'extra_tree_classifier',
    'gaussian_process_classifier',
    'hist_gradient_boosting_classifier',
    'linear_discriminant_analysis',
    'linear_svc',
    'logistic_regression_cv',
    'multinomial_nb',
    'nearest_centroid',
    'nu_svc',
    'passive_aggressive_classifier',
    'perceptron',
    'quadratic_discriminant_analysis',
    'radius_neighbors_classifier',
    'ridge_classifier',
    'sgd_classifier',
}

SEMI_SUPERVISED_IDS = {
    'self_training_classifier',
    'label_propagation',
    'label_spreading',
}

REGRESSION_IDS = {
    'linear_regression',
    'ridge_regression',
    'random_forest_regressor',
    'decision_tree_regressor',
    'extra_trees_regressor',
    'k_neighbors_regressor',
    'lasso_regression',
    'elastic_net_regression',
    'svm_regressor',
    'mlp_regressor',
    'ada_boost_regressor',
    'bagging_regressor',
    'bayesian_ridge',
    'extra_tree_regressor',
    'gamma_regressor',
    'gaussian_process_regressor',
    'gradient_boosting_regressor',
    'hist_gradient_boosting_regressor',
    'huber_regressor',
    'isotonic_regression',
    'kernel_ridge',
    'lars',
    'lasso_cv',
    'linear_svr',
    'nu_svr',
    'passive_aggressive_regressor',
    'poisson_regressor',
    'quantile_regressor',
    'ransac_regressor',
    'radius_neighbors_regressor',
    'sgd_regressor',
    'theil_sen_regressor',
    'tweedie_regressor',
}

CLUSTERING_IDS = {
    'kmeans',
    'mini_batch_kmeans',
    'gaussian_mixture',
    'affinity_propagation',
    'agglomerative_clustering',
    'birch',
    'bisecting_kmeans',
    'dbscan',
    'hdbscan',
    'mean_shift',
    'optics',
    'spectral_clustering',
}
ANOMALY_IDS = {
    'isolation_forest',
    'one_class_svm',
    'local_outlier_factor',
    'elliptic_envelope',
}
RL_IDS = {'fitted_q_iteration'}

FEATURE_COLUMNS = ['feature1', 'feature2', 'feature3', 'colour']


def dataset(task_type: str, n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Mixed numeric/categorical dataset mirroring real uploads."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    df = pd.DataFrame(
        {
            'feature1': x[:, 0],
            'feature2': x[:, 1],
            'feature3': x[:, 2],
            'colour': rng.choice(['red', 'green', 'blue'], size=n),
        }
    )
    if task_type == 'classification':
        df['label'] = np.where(x[:, 0] + x[:, 1] > 0.0, 'yes', 'no')
    elif task_type == 'semi_supervised':
        labels = np.where(x[:, 0] + x[:, 1] > 0.0, 'yes', 'no').astype(object)
        labels[::4] = ''  # empty cells are unlabelled
        df['label'] = labels
    elif task_type == 'regression':
        df['label'] = 2.5 * x[:, 0] - x[:, 1] + rng.normal(0.0, 0.1, n)
    return df


def job(
    algorithm_id: str,
    task_type: str,
    hyperparameters: dict | None = None,
) -> Job:
    """Transient Job instance — enough for `train_pipeline` (no DB flush)."""
    return Job(
        feature_columns=FEATURE_COLUMNS,
        label_column=(
            'label' if task_type in {'classification', 'regression', 'semi_supervised'} else None
        ),
        task_type=task_type,
        algorithm_id=algorithm_id,
        split_ratio=0.25,
        random_seed=42,
        hyperparameters=hyperparameters or {},
    )


def onnx_feeds(df: pd.DataFrame) -> dict[str, np.ndarray]:
    feeds = {}
    for col in FEATURE_COLUMNS:
        values = df[col]
        if pd.api.types.is_numeric_dtype(values):
            feeds[col] = values.to_numpy(dtype=np.float32).reshape(-1, 1)
        else:
            feeds[col] = values.astype(str).to_numpy(dtype=object).reshape(-1, 1)
    return feeds


def run_onnx(onnx_bytes: bytes, df: pd.DataFrame) -> np.ndarray:
    session = ort.InferenceSession(onnx_bytes, providers=['CPUExecutionProvider'])
    outputs = session.run(None, onnx_feeds(df))
    names = [output.name for output in session.get_outputs()]
    predicted = outputs[names.index('label')] if 'label' in names else outputs[0]
    return np.asarray(predicted).ravel()


# --- Reinforcement learning (FEATURE-11) -------------------------------------

RL_FEATURES = ['f1', 'f2']
RL_COLUMNS = {
    'actionColumn': 'action',
    'rewardColumn': 'reward',
    'nextStateColumns': ['nf1', 'nf2'],
    'doneColumn': 'done',
    'episodeColumn': 'episode',
}


def rl_dataset(episodes: int = 8, steps: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(episodes):
        state = rng.normal(size=2)
        for step in range(steps):
            action = rng.choice(['left', 'stay', 'right'])
            reward = float(rng.normal())
            nxt = state + rng.normal(scale=0.1, size=2)
            rows.append(
                {
                    'f1': state[0],
                    'f2': state[1],
                    'action': action,
                    'reward': reward,
                    'nf1': nxt[0],
                    'nf2': nxt[1],
                    'episode': episode,
                    'done': step == steps - 1,
                }
            )
            state = nxt
    return pd.DataFrame(rows)


def rl_job() -> Job:
    return Job(
        feature_columns=RL_FEATURES,
        label_column=None,
        task_type='reinforcement_learning',
        algorithm_id='fitted_q_iteration',
        rl_columns=RL_COLUMNS,
        split_ratio=0.25,
        random_seed=42,
        hyperparameters={'n_iterations': 3, 'gamma': 0.9, 'max_depth': 3, 'learning_rate': 0.1},
    )
