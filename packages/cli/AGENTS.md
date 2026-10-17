# AGENTS.md — CLI (`packages/cli`)

`skl` is the scikit-learn-client CLI. `skl run` trains a scikit-learn
model from a CSV and exports an artefact (ONNX / PKL / JobLib) entirely
offline — no backend required. Every input (file location, features, label,
algorithm, output location) can be passed as flags or interactively.

## Specification

- [`../../docs/11-deployment/cli.md`](../../docs/11-deployment/cli.md) —
  CLI documentation: install, usage, flags.
- [`../../docs/02-features/FEATURE-01-Dataset-Upload.md`](../../docs/02-features/FEATURE-01-Dataset-Upload.md)
- [`../../docs/02-features/FEATURE-02-Column-Mapping.md`](../../docs/02-features/FEATURE-02-Column-Mapping.md)
- [`../../docs/02-features/FEATURE-03-Model-Selection.md`](../../docs/02-features/FEATURE-03-Model-Selection.md)
- [`../../docs/02-features/FEATURE-04-Training-Job.md`](../../docs/02-features/FEATURE-04-Training-Job.md)
- [`../../docs/02-features/FEATURE-05-Model-Export.md`](../../docs/02-features/FEATURE-05-Model-Export.md)

## Layout

```
src/skl/
├── __init__.py            # package entry, exposes main()
├── cli.py                 # click group: run (offline)
├── helpers.py             # inquirer prompts, tabulate display, formatting
├── local.py               # offline train+export pipeline (no backend)
├── pipeline.py            # sklearn training pipelines per task type
├── estimator.py           # estimator construction
├── export.py              # ONNX / pickle / joblib export
├── preprocessor.py        # sklearn ColumnTransformer preprocessor
├── pyinstaller_entry.py   # frozen entry point (package context preserved)
└── catalogue/             # algorithm catalogue (mirrors the backend)
    ├── __init__.py        # ALGORITHMS, TASK_TYPES, LABEL_REQUIRED_TASKS
    ├── algorithms.py
    ├── classification.py
    ├── regression.py
    └── unsupervised.py

skl.spec                   # PyInstaller spec (reference; CI/build use the entry point)
scripts/fix_metadata.py    # importlib.metadata shim for the frozen bundle
scripts/install.sh         # downloads the latest release binary
```

## Run

```sh
# From the CLI directory
uv run skl --help
uv run skl run examples/csv/demo.csv --features x --label y \
  --task-type regression --algorithm linear_regression \
  --format onnx --output examples/models/demo.onnx --no-interactive
```

Flags may be omitted to be prompted interactively (`inquirer`):

```sh
uv run skl run examples/csv/demo.csv   # fully interactive
uv run skl run data/train.csv -t classification -f a,b -l target -a random_forest_classifier -o ./out
```

Build a standalone binary with PyInstaller:

```sh
make build        # produces dist/skl (single-file executable)
./dist/skl run examples/csv/demo.csv --features x --label y \
  --task-type regression --algorithm linear_regression --output examples/models/demo.onnx --no-interactive
```

## Examples

`examples/examples.sh` runs the built binary against
`examples/csv/demo.csv` and writes `examples/models/demo.onnx`:

```sh
./examples/examples.sh
./examples/examples.sh /path/to/custom/skl
```

## Environment

`skl` reads no environment variables. It is fully offline: the `run` command
never contacts a backend.

## Code style

- Python 3.14.8
- `click` for CLI framing, `inquirer` for interactive prompts
- Keep functions small; each helper does one thing
