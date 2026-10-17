"""Unit tests for src/training/core/preprocessor.py helper functions."""

import pandas as pd
from src.training.core.preprocessor import build_preprocessor


def test_build_preprocessor_numeric_only():
    """Test build_preprocessor with only numeric columns."""
    df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    feature_columns = ['feature1', 'feature2']
    preprocessor = build_preprocessor(df, feature_columns)

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 1
    assert preprocessor.transformers[0][0] == 'num'


def test_build_preprocessor_categorical_only():
    """Test build_preprocessor with only categorical columns."""
    df = pd.DataFrame({'cat1': ['a', 'b', 'c'], 'cat2': ['x', 'y', 'z']})
    feature_columns = ['cat1', 'cat2']
    preprocessor = build_preprocessor(df, feature_columns)

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 1
    assert preprocessor.transformers[0][0] == 'cat'


def test_build_preprocessor_mixed():
    """Test build_preprocessor with mixed numeric and categorical columns."""
    df = pd.DataFrame(
        {
            'num1': [1, 2, 3],
            'cat1': ['a', 'b', 'c'],
            'num2': [4, 5, 6],
        }
    )
    feature_columns = ['num1', 'cat1', 'num2']
    preprocessor = build_preprocessor(df, feature_columns)

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2
    transformer_names = [t[0] for t in preprocessor.transformers]
    assert 'num' in transformer_names
    assert 'cat' in transformer_names


def test_build_preprocessor_empty_feature_columns():
    """Test build_preprocessor with empty feature columns."""
    df = pd.DataFrame({'feature1': [1, 2, 3]})
    feature_columns = []
    preprocessor = build_preprocessor(df, feature_columns)

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 0


def test_build_preprocessor_handles_numeric_detection():
    """Test that build_preprocessor correctly detects numeric dtypes."""
    df = pd.DataFrame(
        {
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c'],
        }
    )
    feature_columns = ['int_col', 'float_col', 'str_col']
    preprocessor = build_preprocessor(df, feature_columns)

    transformer_map = {t[0]: t for t in preprocessor.transformers}
    assert 'num' in transformer_map
    assert 'cat' in transformer_map
    assert len(transformer_map['num'][2]) == 2  # int_col and float_col
    assert len(transformer_map['cat'][2]) == 1  # str_col
