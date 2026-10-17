"""Offline reinforcement learning: Fitted Q-Iteration (FEATURE-11).

The public entry point is :func:`train_reinforcement_learning`, which turns an
experience CSV into a fitted Q-function, the documented metrics, and chart data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.logging.config import get_logger
from src.training.rl.transitions import _split, _states_frame, prepare_transitions
from src.training.rl.utils import RLPipeline

logger = get_logger('skl.rl')
MAX_BINS = 50


def _bellman_target(
    row: pd.Series,
    preprocessor: Any,
    q_function: Any,
    actions: list,
    gamma: float,
    feature_columns: list[str],
) -> float:
    r = float(row['r'])
    if bool(row['done']) or row['s_next'] is None:
        return r
    z_next = preprocessor.transform(_states_frame([row['s_next']], feature_columns))[0]
    best = -np.inf
    for action in actions:
        oh = np.zeros(len(actions))
        oh[actions.index(action)] = 1.0
        value = float(q_function.predict(np.concatenate([z_next, oh]).reshape(1, -1))[0])
        best = max(best, value)
    return r + gamma * best


def _design_matrix(
    trans: pd.DataFrame,
    preprocessor: Any,
    actions: list,
    feature_columns: list[str],
) -> np.ndarray:
    rows = []
    for _, row in trans.iterrows():
        z = preprocessor.transform(_states_frame([row['s']], feature_columns))[0]
        oh = np.zeros(len(actions))
        oh[actions.index(row['a'])] = 1.0
        rows.append(np.concatenate([z, oh]))
    return np.asarray(rows)


def train_reinforcement_learning(
    job,
    df: pd.DataFrame,
    pipeline: Pipeline,
    seed: int,
    split_ratio: float,
) -> tuple[RLPipeline, dict[str, Any], dict[str, Any]]:
    """Fit FQI and return ``(RLPipeline, metrics, charts)``."""
    rl = job.rl_columns or {}
    if job.label_column:
        raise ValueError(
            'INVALID_COLUMN_MAP: reinforcement_learning jobs must not set a label column'
        )

    trans, actions = prepare_transitions(df, job)
    episode_col = rl.get('episodeColumn')
    train_df, test_df = _split(trans, episode_col, split_ratio, seed)

    hp = job.hyperparameters or {}
    n_iterations = int(hp.get('n_iterations', 10))
    gamma = float(hp.get('gamma', 0.9))
    max_depth = int(hp.get('max_depth', 3))
    learning_rate = float(hp.get('learning_rate', 0.1))

    preprocessor = pipeline.named_steps['preprocessor']
    preprocessor.fit(_states_frame(list(train_df['s']), job.feature_columns))

    q_function: Any = None
    progress: list[dict[str, float | int]] = []
    for iteration in range(n_iterations):
        # Iteration 0 builds targets from Q0(s, a) = 0; later sweeps use the
        # Q-function fitted in the previous sweep.
        fitted = iteration > 0
        X = _design_matrix(train_df, preprocessor, actions, job.feature_columns)
        if fitted:
            y = np.array(
                [
                    _bellman_target(
                        row, preprocessor, q_function, actions, gamma, job.feature_columns
                    )
                    for _, row in train_df.iterrows()
                ]
            )
        else:
            y = np.array([float(row['r']) for _, row in train_df.iterrows()])
        q_function = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        )
        q_function.fit(X, y)
        progress.append({'iteration': iteration + 1, 'meanTarget': float(np.mean(y))})
        logger.debug(
            'FQI iteration %d/%d mean target %.4f',
            iteration + 1,
            n_iterations,
            float(np.mean(y)),
        )

    rl_pipeline = RLPipeline(preprocessor, q_function, actions, list(job.feature_columns))

    q_values: list[float] = []
    td_errors: list[float] = []
    rewards: list[float] = []
    for _, row in test_df.iterrows():
        q = rl_pipeline._q_value(row['s'], row['a'])
        target = _bellman_target(row, preprocessor, q_function, actions, gamma, job.feature_columns)
        q_values.append(q)
        td_errors.append(target - q)
        rewards.append(float(row['r']))

    metrics: dict[str, Any] = {
        'meanReward': float(np.mean(rewards)) if rewards else 0.0,
        'meanQ': float(np.mean(q_values)) if q_values else 0.0,
        'tdErrorMean': float(np.mean(np.abs(td_errors))) if td_errors else 0.0,
        'tdErrorStd': float(np.std(td_errors)) if td_errors else 0.0,
        'actions': [str(a) for a in actions],
    }

    charts: dict[str, Any] = {
        'qValueHistogram': _histogram(q_values),
        'tdErrorHistogram': _histogram(td_errors),
        'fqiProgress': progress,
    }

    if episode_col and 'episode' in test_df.columns:
        metrics['episodeCount'] = int(test_df['episode'].nunique())
        returns = [float(g['r'].sum()) for _, g in test_df.groupby('episode')]
        if returns:
            metrics['meanEpisodeReturn'] = float(np.mean(returns))
        charts['episodeReturns'] = [
            {'episode': str(ep), 'return': float(g['r'].sum())}
            for ep, g in test_df.groupby('episode')
        ]

    return rl_pipeline, metrics, charts


def _histogram(values: list[float]) -> dict[str, list[float]]:
    if not values:
        return {'edges': [], 'counts': []}
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=MAX_BINS)
    return {'edges': edges.tolist(), 'counts': counts.tolist()}
