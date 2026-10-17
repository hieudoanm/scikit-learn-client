"""Reinforcement learning utility classes and helpers."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def _is_truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, float) and np.isnan(val):
        return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {'true', '1', 'yes', 'y'}:
            return True
        if s in {'false', '0', 'no', 'n', ''}:
            return False
    try:
        if pd.isna(val):
            return False
    except TypeError, ValueError:
        pass
    try:
        return float(val) != 0.0
    except TypeError, ValueError:
        return bool(val)


class PositionalPreprocessor(BaseEstimator, TransformerMixin):
    """Adapts a named-column transformer to positional float input.

    ONNX inputs are plain float tensors, while the shared preprocessor selects
    columns by name. This wrapper reconstructs the frame from the tensor and
    returns the transformed matrix.
    """

    def __init__(self, preprocessor: Any, feature_columns: list[str]) -> None:
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns

    def fit(self, X, y=None):  # pragma: no cover - never refit at export
        return self

    def transform(self, X):
        frame = pd.DataFrame(np.asarray(X), columns=self.feature_columns)
        return np.asarray(self.preprocessor.transform(frame))

    def get_feature_names_out(self, input_features=None):  # pragma: no cover
        return self.preprocessor.get_feature_names_out(input_features)


class RLPipeline(BaseEstimator, TransformerMixin):
    """Preprocessor + fitted Q-function; predicts the greedy action.

    The state block is passed through the shared preprocessor, the action is
    one-hot encoded, and the concatenation is scored by the Q regressor.
    """

    def __init__(
        self,
        preprocessor: Any,
        q_function: Any,
        actions: list,
        feature_columns: list[str],
    ) -> None:
        self.preprocessor = preprocessor
        self.q_function = q_function
        self.actions = actions
        self.feature_columns = feature_columns

    def _q_value(self, state_row: np.ndarray, action: Any) -> float:
        frame = pd.DataFrame([state_row], columns=self.feature_columns)
        z = self.preprocessor.transform(frame)[0]
        oh = np.zeros(len(self.actions))
        oh[self.actions.index(action)] = 1.0
        x = np.concatenate([z, oh]).reshape(1, -1)
        return float(self.q_function.predict(x)[0])

    def q_values(self, state_row: np.ndarray) -> list[float]:
        return [self._q_value(state_row, action) for action in self.actions]

    def greedy_action(self, state_row: np.ndarray) -> Any:
        values = self.q_values(state_row)
        return self.actions[int(np.argmax(values))]

    def predict(self, X: Any) -> np.ndarray:
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=float)
        return np.array([self.greedy_action(X[i]) for i in range(X.shape[0])])

    def export_estimator(self) -> Pipeline:
        """Return a pipeline mapping ``[state, onehot_action]`` -> Q-value."""
        return Pipeline(steps=[('q_function', self.q_function)])

    def build_onnx(self, initial_types):
        """Build an ONNX model with the FEATURE-11 input contract.

        ``initial_types`` declares the ``state`` and ``action`` inputs. The
        graph standardises the raw state block with the fitted scaler, one-hot
        concatenates the action block, and scores the Q regressor, emitting a
        single scalar Q-value per row.
        """
        import onnx
        from onnx import TensorProto, helper
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        n_features = len(self.feature_columns)
        n_actions = len(self.actions)

        mean, scale = self._standardizer()
        q_model = convert_sklearn(
            self.export_estimator(),
            initial_types=[('q_input', FloatTensorType([None, n_features + n_actions]))],
            target_opset={'': 15, 'ai.onnx.ml': 3},
        )

        # Standardise the state tensor: (state - mean) / scale.
        sub = helper.make_node(
            'Sub',
            inputs=['state', 'scaler_mean'],
            outputs=['state_centered'],
        )
        div = helper.make_node(
            'Div',
            inputs=['state_centered', 'scaler_scale'],
            outputs=['state_scaled'],
        )
        concat = helper.make_node(
            'Concat',
            inputs=['state_scaled', 'action'],
            outputs=[self._q_input_name(q_model)],
            axis=1,
        )

        mean_init = helper.make_tensor(
            'scaler_mean', TensorProto.FLOAT, [n_features], mean.astype(np.float32).ravel()
        )
        scale_init = helper.make_tensor(
            'scaler_scale', TensorProto.FLOAT, [n_features], scale.astype(np.float32).ravel()
        )

        state_input = helper.make_tensor_value_info('state', TensorProto.FLOAT, [None, n_features])
        action_input = helper.make_tensor_value_info('action', TensorProto.FLOAT, [None, n_actions])
        q_output = list(q_model.graph.output)[0]
        output = helper.make_tensor_value_info(q_output.name, TensorProto.FLOAT, [None, 1])

        graph = helper.make_graph(
            nodes=[sub, div, concat, *q_model.graph.node],
            name='skl-rl-q',
            inputs=[state_input, action_input],
            outputs=[output],
            initializer=[mean_init, scale_init, *q_model.graph.initializer],
        )
        model = helper.make_model(
            graph,
            opset_imports=list(q_model.opset_import),
            producer_name='skl-rl',
        )
        model.ir_version = q_model.ir_version
        onnx.checker.check_model(model)
        return model

    def _q_input_name(self, q_model) -> str:
        return q_model.graph.input[0].name

    def _standardizer(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (mean, scale) used to standardise the state features."""
        scaler = self.preprocessor.named_transformers_.get('num')
        if scaler is None or not hasattr(scaler, 'mean_'):
            n = len(self.feature_columns)
            return np.zeros(n), np.ones(n)
        return np.asarray(scaler.mean_, dtype=float), np.asarray(scaler.scale_, dtype=float)
