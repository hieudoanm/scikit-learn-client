"""Transition preparation for reinforcement learning."""

from typing import Any

import numpy as np
import pandas as pd

from src.training.rl.utils import _is_truthy

MAX_POINTS = 500
MIN_TRANSITIONS = 10


def prepare_transitions(df: pd.DataFrame, job) -> tuple[pd.DataFrame, list]:
    """Resolve each row into a transition ``(s, a, r, s', done, episode)``."""
    rl = job.rl_columns or {}
    for key in ('actionColumn', 'rewardColumn'):
        if key not in rl:
            raise ValueError(f'INVALID_COLUMN_MAP: rlColumns.{key} is required')

    action_col = rl['actionColumn']
    reward_col = rl['rewardColumn']
    done_col = rl.get('doneColumn')
    episode_col = rl.get('episodeColumn')
    next_state_cols = rl.get('nextStateColumns')

    for col in (action_col, reward_col, done_col, episode_col):
        if col and col not in df.columns:
            raise ValueError(f'INVALID_COLUMN_MAP: unknown experience column: {col}')
        if col and col in job.feature_columns:
            raise ValueError(
                f'INVALID_COLUMN_MAP: experience column "{col}" is also a state feature'
            )

    if next_state_cols:
        if len(next_state_cols) != len(job.feature_columns):
            raise ValueError(
                'INVALID_COLUMN_MAP: nextStateColumns must mirror featureColumns in length '
                'and order'
            )
        for col in next_state_cols:
            if col not in df.columns:
                raise ValueError(f'INVALID_COLUMN_MAP: unknown experience column: {col}')
            if col in job.feature_columns:
                raise ValueError(
                    f'INVALID_COLUMN_MAP: experience column "{col}" is also a state feature'
                )
    elif not episode_col:
        raise ValueError('INVALID_COLUMN_MAP: nextStateColumns or episodeColumn is required')

    if not pd.api.types.is_numeric_dtype(df[reward_col]):
        try:
            pd.to_numeric(df[reward_col])
        except (TypeError, ValueError) as exc:
            raise ValueError('INVALID_LABEL_TYPE: reward column must be numeric') from exc

    actions = sorted(df[action_col].dropna().unique().tolist())
    if not 2 <= len(actions) <= 50:
        raise ValueError('INVALID_COLUMN_MAP: action column must have between 2 and 50 values')

    next_states: list[Any] = [None] * len(df)
    last_in_episode: set = set()
    if next_state_cols:
        for pos, (_, row) in enumerate(df.iterrows()):
            next_states[pos] = row[next_state_cols].to_numpy(dtype=float)
    else:
        positions = {idx: pos for pos, idx in enumerate(df.index)}
        grouped = df.groupby(episode_col, sort=False)
        for _, group in grouped:
            members = list(group.index)
            last_in_episode.add(members[-1])
            for i in range(len(members) - 1):
                cur = positions[members[i]]
                nxt = df.loc[members[i + 1], job.feature_columns].to_numpy(dtype=float)
                next_states[cur] = nxt

    rows = []
    for pos, (idx, row) in enumerate(df.iterrows()):
        s = row[job.feature_columns].to_numpy(dtype=float)
        s_next = next_states[pos]
        done = _is_truthy(row[done_col]) if done_col else False
        if s_next is None or idx in last_in_episode:
            done = True
        rows.append(
            {
                's': s,
                'a': row[action_col],
                'r': float(row[reward_col]),
                's_next': s_next,
                'done': done,
                'episode': row[episode_col] if episode_col else None,
            }
        )

    trans = pd.DataFrame(rows)
    if len(trans) < MIN_TRANSITIONS:
        raise ValueError(
            f'INVALID_HYPERPARAMETERS: at least {MIN_TRANSITIONS} transitions are required'
        )
    if episode_col and trans['episode'].nunique() < 2:
        raise ValueError('INVALID_HYPERPARAMETERS: at least two episodes are required')
    return trans, actions


def _split(trans: pd.DataFrame, episode_col: str | None, split_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    if episode_col and 'episode' in trans.columns:
        episodes = np.array(trans['episode'].unique())
        rng.shuffle(episodes)
        n_hold = max(1, int(round(len(episodes) * split_ratio)))
        hold = set(episodes[:n_hold].tolist())
        train_mask = ~trans['episode'].isin(hold)
        test_mask = trans['episode'].isin(hold)
    else:
        order = rng.permutation(len(trans))
        n_hold = max(1, int(round(len(trans) * split_ratio)))
        hold = set(order[:n_hold].tolist())
        mask = np.zeros(len(trans), dtype=bool)
        mask[list(hold)] = True
        train_mask = ~mask
        test_mask = mask
    return (
        trans[train_mask].reset_index(drop=True),
        trans[test_mask].reset_index(drop=True),
    )


def _states_frame(states: list[np.ndarray], feature_columns: list[str]) -> pd.DataFrame:
    matrix = np.vstack([np.asarray(s, dtype=float).reshape(1, -1) for s in states])
    return pd.DataFrame(matrix, columns=feature_columns)
