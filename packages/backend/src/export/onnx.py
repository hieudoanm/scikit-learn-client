"""Model export utilities: ONNX and joblib."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd

from src.storage.filesystem import write_artefact


def _patch_ml_dtypes() -> None:
    """Add missing float types to ml_dtypes for onnx compatibility.

    onnx>=1.13 references ml_dtypes attributes (float4_e2m1fn, float8_e4m3fnuz,
    float8_e5m2fnuz, uint4, int4, float8_e8m0fnu) that are missing from
    all published ml_dtypes versions. Patch them in before importing onnx.
    """
    try:
        import ml_dtypes

        if not hasattr(ml_dtypes, 'float4_e2m1fn'):
            ml_dtypes.float4_e2m1fn = np.dtype(np.uint8)
        if not hasattr(ml_dtypes, 'float4_e2m1fn_x2'):
            ml_dtypes.float4_e2m1fn_x2 = np.dtype(np.uint8)
        if not hasattr(ml_dtypes, 'float8_e4m3fnuz'):
            ml_dtypes.float8_e4m3fnuz = np.dtype(np.float32)
        if not hasattr(ml_dtypes, 'float8_e5m2fnuz'):
            ml_dtypes.float8_e5m2fnuz = np.dtype(np.float32)
        if not hasattr(ml_dtypes, 'float8_e8m0fnu'):
            ml_dtypes.float8_e8m0fnu = np.dtype(np.uint8)
        if not hasattr(ml_dtypes, 'uint4'):
            ml_dtypes.uint4 = np.dtype(np.uint8)
        if not hasattr(ml_dtypes, 'int4'):
            ml_dtypes.int4 = np.dtype(np.int8)
    except ImportError:
        pass


def _get_skl2onnx():
    """Lazily import skl2onnx; raises a clear error if the onnx extra is missing."""
    _patch_ml_dtypes()
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType

        return convert_sklearn, FloatTensorType, StringTensorType
    except ImportError as exc:  # pragma: no cover
        raise ImportError("onnx export requires the 'onnx' extra: uv sync --extra onnx") from exc


def _infer_initial_types(
    FloatTensorType, StringTensorType, df: pd.DataFrame, feature_columns: list[str]
):
    """Build per-column initial types for ONNX conversion.

    ColumnTransformer needs named column inputs, so we specify each column
    with its inferred ONNX type.
    """
    initial_types = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            initial_types.append((col, FloatTensorType([None, 1])))
        else:
            initial_types.append((col, StringTensorType([None, 1])))
    return initial_types


def export_onnx(pipeline, X_sample: pd.DataFrame, job_id) -> tuple[bytes, str, str, int]:
    """Export a fitted sklearn pipeline to ONNX bytes.

    Returns (onnx_bytes, filename, storage_key, byte_size).
    """
    convert_sklearn, FloatTensorType, StringTensorType = _get_skl2onnx()
    feature_columns = list(X_sample.columns)
    initial_types = _infer_initial_types(
        FloatTensorType, StringTensorType, X_sample, feature_columns
    )
    # Pin the ML domain to 3: IsolationForest otherwise emits operators from
    # ai.onnx.ml v4, which the pinned ONNX Runtime rejects.
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset={'': 15, 'ai.onnx.ml': 3},
    )
    onnx_bytes = onnx_model.SerializeToString()

    filename = f'{job_id}.onnx'
    storage_key, path, byte_size = write_artefact(job_id, onnx_bytes, filename)
    return onnx_bytes, filename, storage_key, byte_size


def export_onnx_rl(pipeline, X_sample: pd.DataFrame, actions: list, job_id):
    """Export an RL Q-function to ONNX.

    Contract (FEATURE-11): inputs are the state features in ``featureColumns``
    order followed by a one-hot action block of length ``len(actions)``; the
    output is one scalar Q-value. Returns (onnx_bytes, filename, key, size).
    """
    from skl2onnx.common.data_types import FloatTensorType

    from src.training.rl import RLPipeline

    if not isinstance(pipeline, RLPipeline):  # pragma: no cover - defensive
        raise ValueError('export_onnx_rl requires an RLPipeline')

    n_features = len(X_sample.columns)
    n_actions = len(actions)
    initial_types = [
        ('state', FloatTensorType([None, n_features])),
        ('action', FloatTensorType([None, n_actions])),
    ]
    onnx_model = pipeline.build_onnx(initial_types)
    onnx_bytes = onnx_model.SerializeToString()

    filename = f'{job_id}.onnx'
    storage_key, path, byte_size = write_artefact(job_id, onnx_bytes, filename)
    return onnx_bytes, filename, storage_key, byte_size


def export_joblib(pipeline, job_id) -> tuple[bytes, str, str, int]:
    """Export a fitted pipeline to joblib bytes.

    Returns (joblib_bytes, filename, storage_key, byte_size).
    """
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    data = buf.getvalue()
    filename = f'{job_id}.joblib'
    storage_key, path, byte_size = write_artefact(job_id, data, filename)
    return data, filename, storage_key, byte_size


def export_pickle(pipeline, job_id) -> tuple[bytes, str, str, int]:
    """Export a fitted pipeline to pickle bytes.

    Returns (pickle_bytes, filename, storage_key, byte_size).
    """
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    data = buf.getvalue()
    filename = f'{job_id}.pkl'
    storage_key, path, byte_size = write_artefact(job_id, data, filename)
    return data, filename, storage_key, byte_size
