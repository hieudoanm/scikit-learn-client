"""Allow-listed sklearn algorithm catalogue for skl CLI."""

from .classification import CLASSIFICATION_ALGORITHMS
from .regression import REGRESSION_ALGORITHMS
from .unsupervised import UNSUPERVISED_ALGORITHMS

ALGORITHMS: dict[str, dict] = {
    **CLASSIFICATION_ALGORITHMS,
    **REGRESSION_ALGORITHMS,
    **UNSUPERVISED_ALGORITHMS,
}

__all__ = ["ALGORITHMS"]
