"""Unit tests for src/skl/preprocessor.py."""

from __future__ import annotations

import pandas as pd

from skl.preprocessor import (
    build_preprocessor,
    labelled_mask,
    semi_supervised_problem,
    semi_supervised_stats,
)


class TestBuildPreprocessor:
    def test_numeric_columns_only(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        pre = build_preprocessor(df, ["a", "b"])
        assert pre is not None
        assert len(pre.transformers) == 1

    def test_categorical_columns_only(self):
        df = pd.DataFrame({"a": ["x", "y", "x"], "b": ["p", "q", "p"]})
        pre = build_preprocessor(df, ["a", "b"])
        assert pre is not None

    def test_mixed_columns(self):
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
        pre = build_preprocessor(df, ["num", "cat"])
        assert pre is not None
        assert len(pre.transformers) == 2


class TestLabelledMask:
    def test_numeric_non_null(self):
        s = pd.Series([1.0, 2.0, None, 4.0])
        mask = labelled_mask(s)
        assert list(mask) == [True, True, False, True]

    def test_string_tokens(self):
        s = pd.Series(["a", "", "nan", "null", "b"])
        mask = labelled_mask(s)
        assert list(mask) == [True, False, False, False, True]

    def test_mixed_case_tokens(self):
        s = pd.Series(["NAN", "Null", "  ", "x"])
        mask = labelled_mask(s)
        assert list(mask) == [False, False, False, True]


class TestSemiSupervisedStats:
    def test_stats(self):
        s = pd.Series([0, 1, None, "", "a", 0, 1])
        stats = semi_supervised_stats(s)
        assert stats["labelledRows"] == 5
        assert stats["unlabelledRows"] == 2
        assert stats["classes"] == 3
        assert stats["smallestClass"] == 1


class TestSemiSupervisedProblem:
    def test_none_when_valid(self):
        assert (
            semi_supervised_problem({"classes": 2, "smallestClass": 5, "labelledRows": 20}) is None
        )

    def test_too_few_classes(self):
        assert (
            semi_supervised_problem({"classes": 1, "smallestClass": 5, "labelledRows": 20})
            is not None
        )

    def test_too_few_per_class(self):
        assert (
            semi_supervised_problem({"classes": 2, "smallestClass": 1, "labelledRows": 20})
            is not None
        )

    def test_too_few_labelled_rows(self):
        assert (
            semi_supervised_problem({"classes": 2, "smallestClass": 5, "labelledRows": 5})
            is not None
        )
