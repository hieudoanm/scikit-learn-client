"""ONNX and joblib export utilities for skl CLI."""

from __future__ import annotations

import io

import pandas as pd


def _patch_ml_dtypes() -> None:
    """Add missing float types to ml_dtypes for onnx compatibility."""
    try:
        import ml_dtypes
        import numpy as np

        if not hasattr(ml_dtypes, "float4_e2m1fn"):
            ml_dtypes.float4_e2m1fn = np.dtype(np.uint8)
        if not hasattr(ml_dtypes, "float4_e2m1fn_x2"):
            ml_dtypes.float4_e2m1fn_x2 = np.dtype(np.uint8)
        if not hasattr(ml_dtypes, "float8_e4m3fnuz"):
            ml_dtypes.float8_e4m3fnuz = np.dtype(np.float32)
        if not hasattr(ml_dtypes, "float8_e5m2fnuz"):
            ml_dtypes.float8_e5m2fnuz = np.dtype(np.float32)
        if not hasattr(ml_dtypes, "float8_e8m0fnu"):
            ml_dtypes.float8_e8m0fnu = np.dtype(np.uint8)
        if not hasattr(ml_dtypes, "uint4"):
            ml_dtypes.uint4 = np.dtype(np.uint8)
        if not hasattr(ml_dtypes, "int4"):
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
    except ImportError as exc:
        raise ImportError("onnx export requires the 'onnx' extra: uv sync --extra onnx") from exc


def _infer_initial_types(FloatTensorType, StringTensorType, df, feature_columns):
    """Build per-column initial types for ONNX conversion."""
    import pandas as pd

    initial_types = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            initial_types.append((col, FloatTensorType([None, 1])))
        else:
            initial_types.append((col, StringTensorType([None, 1])))
    return initial_types


def export_onnx(pipeline, X_sample: pd.DataFrame) -> bytes:
    """Export a fitted sklearn pipeline to ONNX bytes."""
    convert_sklearn, FloatTensorType, StringTensorType = _get_skl2onnx()
    feature_columns = list(X_sample.columns)
    initial_types = _infer_initial_types(
        FloatTensorType, StringTensorType, X_sample, feature_columns
    )
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=16,
    )
    return onnx_model.SerializeToString()


def export_pickle(pipeline) -> bytes:
    """Export a fitted sklearn pipeline to pickle bytes."""
    import joblib

    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    return buf.getvalue()


def export_joblib(pipeline) -> bytes:
    """Export a fitted sklearn pipeline to joblib bytes."""
    import joblib

    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    return buf.getvalue()
