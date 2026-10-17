"""Unit tests for src/skl/estimator.py."""

from __future__ import annotations

import pytest

from skl.estimator import build_export_pipeline, make_estimator


class TestMakeEstimator:
    def test_known_algorithm(self):
        est = make_estimator("linear_regression", {}, seed=42)
        assert est is not None

    def test_random_state_injected(self):
        est = make_estimator("ridge_regression", {"alpha": 1.0}, seed=7)
        assert est.random_state == 7

    def test_hyperparameters_applied(self):
        est = make_estimator("ridge_regression", {"alpha": 2.5}, seed=42)
        assert est.alpha == 2.5

    def test_unknown_hyperparameter_raises(self):
        with pytest.raises(ValueError, match="INVALID_HYPERPARAMETERS"):
            make_estimator("linear_regression", {"bogus_param": 1}, seed=42)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(KeyError):
            make_estimator("does_not_exist", {}, seed=42)


class TestBuildExportPipeline:
    def test_passthrough(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("scaler", StandardScaler())])
        assert build_export_pipeline(pipe) is pipe

    def test_self_training_unwraps(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.semi_supervised import SelfTrainingClassifier

        inner = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=100))])
        stc = SelfTrainingClassifier(LogisticRegression(max_iter=100))
        pipe = Pipeline([("pre", inner), ("estimator", stc)])
        result = build_export_pipeline(pipe)
        assert result is not pipe
        assert isinstance(result.steps[-1][1], type(LogisticRegression()))
