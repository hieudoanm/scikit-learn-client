"""Allow-listed sklearn algorithm catalogue.

Ids mirror the catalogue table in
``docs/01-requirements/functional-requirements.md`` (keep them in sync).
Entries are grouped by task family in :mod:`catalogue_classification`,
:mod:`catalogue_regression`, and :mod:`catalogue_unsupervised`, and merged here
into the flat ``ALGORITHMS`` mapping that :mod:`src.training.catalogue`
re-exports for the rest of the codebase.
"""

from src.training.catalogue.classification import CLASSIFICATION_ALGORITHMS
from src.training.catalogue.regression import REGRESSION_ALGORITHMS
from src.training.catalogue.unsupervised import UNSUPERVISED_ALGORITHMS

ALGORITHMS: dict[str, dict] = {
    **CLASSIFICATION_ALGORITHMS,
    **REGRESSION_ALGORITHMS,
    **UNSUPERVISED_ALGORITHMS,
}
