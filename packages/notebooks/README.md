# Notebooks

Reproducible Jupyter notebooks for skl, plus the sample datasets they export.

| Notebook         | What it does                                                                                              |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| `datasets.ipynb` | Exports the sample datasets in `datasets/` (`regression_diabetes.csv`, `classification_iris.csv`).         |
| `pipeline.ipynb` | Reproduces the backend training pipeline (`packages/backend/src/training/core/pipeline.py`) step by step. |

## Commands

```sh
make lint            # ruff check . --fix
make format          # ruff format .
make run             # execute pipeline.ipynb in place
make export-datasets # execute datasets.ipynb in place
```

Both notebooks are executed with `uv run jupyter nbconvert --execute`, so the
committed `.ipynb` files carry their outputs. Re-running a notebook overwrites
its outputs in place and, for `datasets.ipynb`, regenerates the CSVs it writes.
