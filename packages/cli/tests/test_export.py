"""Unit tests for src/skl/export.py."""

from __future__ import annotations

import io

import onnx
import onnxruntime as ort
import pandas as pd

from skl.export import export_joblib, export_onnx, export_pickle


class TestExportOnnx:
    def test_returns_bytes(self, regression_csv):
        df = pd.read_csv(regression_csv)
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("scaler", StandardScaler()), ("est", LinearRegression())])
        pipe.fit(df[["x"]], df["y"])
        blob = export_onnx(pipe, df[["x"]])
        assert isinstance(blob, bytes)
        assert len(blob) > 0
        model = (
            onnx.LoadFromString(blob)
            if hasattr(onnx, "LoadFromString")
            else onnx.load_from_string(blob)
        )
        assert model is not None

    def test_loadable_by_onnxruntime(self, regression_csv):
        df = pd.read_csv(regression_csv)
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("scaler", StandardScaler()), ("est", LinearRegression())])
        pipe.fit(df[["x"]], df["y"])
        blob = export_onnx(pipe, df[["x"]])
        sess = ort.InferenceSession(blob)
        feed = {i.name: df[[i.name]].to_numpy(dtype="float32") for i in sess.get_inputs()}
        out = sess.run(None, feed)
        assert len(out) == 1


class TestExportPickle:
    def test_returns_bytes(self, regression_csv):
        import joblib

        df = pd.read_csv(regression_csv)
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("scaler", StandardScaler()), ("est", LinearRegression())])
        pipe.fit(df[["x"]], df["y"])
        blob = export_pickle(pipe)
        assert isinstance(blob, bytes)
        loaded = joblib.load(io.BytesIO(blob))
        assert loaded is not None


class TestExportJoblib:
    def test_returns_bytes(self, regression_csv):
        df = pd.read_csv(regression_csv)
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("scaler", StandardScaler()), ("est", LinearRegression())])
        pipe.fit(df[["x"]], df["y"])
        blob = export_joblib(pipe)
        assert isinstance(blob, bytes)
        assert len(blob) > 0
