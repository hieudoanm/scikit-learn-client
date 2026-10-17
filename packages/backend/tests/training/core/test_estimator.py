"""Unit tests for src/training/core/estimator.py (estimator + version helpers)."""

from src.training.core.estimator import library_versions


def test_library_versions():
    """Test that library_versions returns a dict with expected keys."""
    versions = library_versions()
    assert isinstance(versions, dict)
    assert 'scikit-learn' in versions
    assert 'numpy' in versions
    assert 'pandas' in versions
    assert 'joblib' in versions
    assert 'python' in versions
    # ONNX packages may be None if not installed
    assert 'skl2onnx' in versions
    assert 'onnxruntime' in versions


def test_library_versions_values_are_strings():
    """Test that library version values are strings or None."""
    versions = library_versions()
    for _key, value in versions.items():
        assert value is None or isinstance(value, str)
