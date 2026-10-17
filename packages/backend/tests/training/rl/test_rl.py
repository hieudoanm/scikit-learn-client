"""Reinforcement-learning training and ONNX parity tests (FEATURE-11).

These cover the offline Fitted Q-Iteration pipeline, its metrics/charts, and
the ``[state, action]`` ONNX Q-value contract (FR-110..FR-115).
"""

from __future__ import annotations

import uuid

import numpy as np
import onnxruntime as ort
import pytest
from src.export.onnx import export_onnx_rl
from src.training.core.pipeline import train_pipeline
from src.training.core.validation import validate_hyperparameters
from tests.training.fixtures import (
    RL_COLUMNS,
    RL_FEATURES,
)
from tests.training.fixtures import (
    rl_dataset as _rl_dataset,
)
from tests.training.fixtures import (
    rl_job as _rl_job,
)


def test_rl_trains_and_exports_onnx_q_values(monkeypatch) -> None:
    """FR-110..FR-115: FQI metrics, charts, and ONNX Q-value parity with sklearn."""
    from src.training.rl import RLPipeline

    df = _rl_dataset()
    job = _rl_job()

    monkeypatch.setattr(
        'src.export.onnx.write_artefact',
        lambda job_id, data, filename: ('test-key', 'in-memory', len(data)),
    )

    pipeline, metrics, charts = train_pipeline(job, df)

    assert isinstance(pipeline, RLPipeline)
    assert {
        'meanReward',
        'meanQ',
        'tdErrorMean',
        'tdErrorStd',
        'actions',
        'episodeCount',
        'meanEpisodeReturn',
    } <= set(metrics)
    assert metrics['actions'] == ['left', 'right', 'stay']
    assert {'qValueHistogram', 'tdErrorHistogram', 'fqiProgress', 'episodeReturns'} <= set(charts)

    onnx_bytes, filename, _, byte_size = export_onnx_rl(
        pipeline, df[RL_FEATURES], pipeline.actions, uuid.uuid4()
    )
    assert filename.endswith('.onnx')
    assert byte_size == len(onnx_bytes)

    session = ort.InferenceSession(onnx_bytes, providers=['CPUExecutionProvider'])
    assert [i.name for i in session.get_inputs()] == ['state', 'action']

    state = df[RL_FEATURES].head(3).to_numpy(dtype=np.float32)
    onehot = np.zeros((3, len(pipeline.actions)), dtype=np.float32)
    for i in range(len(pipeline.actions)):
        onehot[i, i] = 1.0
    got = session.run(None, {'state': state, 'action': onehot})[0].ravel()
    want = [
        pipeline._q_value(state[i].astype(float), action)
        for i, action in enumerate(pipeline.actions)
    ]
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_rl_validation_errors() -> None:
    """FR-111/FR-112: missing experience columns and bad action sets are rejected."""
    df = _rl_dataset()
    job = _rl_job()
    job.rl_columns = {}

    with pytest.raises(ValueError, match='INVALID_COLUMN_MAP'):
        train_pipeline(job, df)

    job = _rl_job()
    job.rl_columns = {**RL_COLUMNS, 'actionColumn': 'f1'}
    with pytest.raises(ValueError, match='INVALID_COLUMN_MAP'):
        train_pipeline(job, df)

    job = _rl_job()
    df_single = df.copy()
    df_single['action'] = 'left'
    with pytest.raises(ValueError, match='INVALID_COLUMN_MAP'):
        train_pipeline(job, df_single)


def test_rl_hyperparameter_validation() -> None:
    validate_hyperparameters('fitted_q_iteration', {}, row_count=100)
    for bad in (
        {'n_iterations': 0},
        {'n_iterations': 51},
        {'gamma': 1.5},
        {'max_depth': 0},
        {'learning_rate': 0},
    ):
        with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
            validate_hyperparameters('fitted_q_iteration', bad)


def test_rl_episode_derived_next_state_and_split() -> None:
    """Episodes resolve s' from CSV order; the last row of each episode is terminal."""
    from src.training.rl import prepare_transitions

    df = _rl_dataset()
    df = df.drop(columns=['nf1', 'nf2'])
    job = _rl_job()
    job.rl_columns = {
        'actionColumn': 'action',
        'rewardColumn': 'reward',
        'episodeColumn': 'episode',
        'doneColumn': 'done',
    }

    trans, actions = prepare_transitions(df, job)
    assert actions == ['left', 'right', 'stay']
    assert len(trans) == len(df)
    # Every episode's final transition is terminal.
    terminal = trans[trans['done']]
    assert len(terminal) == df['episode'].nunique()
    # Non-terminal transitions carry a next state.
    assert trans[~trans['done']]['s_next'].notna().all()


def test_rl_rejects_too_few_transitions() -> None:
    from src.training.rl import prepare_transitions

    df = _rl_dataset().head(5)
    job = _rl_job()
    with pytest.raises(ValueError, match='INVALID_HYPERPARAMETERS'):
        prepare_transitions(df, job)


def test_rl_label_column_is_rejected() -> None:
    df = _rl_dataset()
    job = _rl_job()
    job.label_column = 'reward'
    with pytest.raises(ValueError, match='INVALID_COLUMN_MAP'):
        train_pipeline(job, df)
