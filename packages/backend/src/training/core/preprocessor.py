"""Data preprocessing utilities."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer preprocessor."""
    numeric_cols = []
    categorical_cols = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    transformers = []
    if numeric_cols:
        transformers.append(('num', StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder='drop')


def labelled_mask(y: pd.Series) -> pd.Series:
    """Return a boolean mask that is true for rows carrying a real label."""
    from src.training.catalogue import _UNLABELLED_TOKENS

    if pd.api.types.is_numeric_dtype(y):
        return y.notna()
    text = y.astype('string').str.strip().str.lower()
    return y.notna() & ~text.isin(_UNLABELLED_TOKENS)


def semi_supervised_stats(y: pd.Series) -> dict[str, int]:
    """Summarise the labelled/unlabelled split of a semi-supervised label column."""
    mask = labelled_mask(y)
    labelled = y[mask]
    counts = labelled.value_counts(dropna=True)
    return {
        'labelledRows': int(mask.sum()),
        'unlabelledRows': int((~mask).sum()),
        'classes': int(labelled.nunique(dropna=True)),
        'smallestClass': int(counts.min()) if len(counts) else 0,
    }


def semi_supervised_problem(stats: dict[str, int]) -> str | None:
    """Return the INSUFFICIENT_LABELS detail, or None when the label set is valid."""
    from src.training.catalogue import (
        MIN_CLASSES,
        MIN_LABELLED_PER_CLASS,
        MIN_LABELLED_ROWS,
    )

    if stats['labelledRows'] < MIN_LABELLED_ROWS:
        return (
            f'At least {MIN_LABELLED_ROWS} labelled rows are required; '
            f'found {stats["labelledRows"]}.'
        )
    if stats['classes'] < MIN_CLASSES:
        return f'At least {MIN_CLASSES} classes are required in the label column.'
    if stats['smallestClass'] < MIN_LABELLED_PER_CLASS:
        return (
            f'Each class needs at least {MIN_LABELLED_PER_CLASS} labelled rows; '
            f'the smallest class has {stats["smallestClass"]}.'
        )
    return None
