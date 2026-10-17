# CLI — skl

Part of [`11-deployment.md`](../11-deployment.md). Documents the `skl`
command-line interface: offline train-and-export from a CSV, binary
installation, and usage.

## Overview

`skl` is the scikit-learn-client CLI. `skl run` trains a scikit-learn model
from a CSV and exports a portable artefact (ONNX / PKL / JobLib) entirely
offline — no backend, no database, no network required. Every input (file
location, features, label, algorithm, output location) can be passed as flags
or supplied interactively.

The CLI is distributed as a single-file PyInstaller binary attached to the
`latest` GitHub Release, and it can also be installed from source with `uv` or
`pip`.

## Installation

### Option 1 — Prebuilt binary (recommended)

Download the `skl` binary from the `latest` release and place it on your PATH:

```bash
curl -fsSL https://github.com/hieudoanm/scikit-learn-client/releases/download/latest/skl \
  -o "$HOME/.local/bin/skl"
chmod +x "$HOME/.local/bin/skl"
```

Ensure `$HOME/.local/bin` is on your `PATH`, then:

```bash
skl --help
```

There is also a maintained installer script in the repository:

```bash
bash packages/cli/scripts/install.sh
```

It downloads the `latest` release binary to `$HOME/.local/bin/skl`.

### Option 2 — From source

```bash
# From packages/cli
uv tool install -e .
# or
pip install -e .
```

### Build a standalone binary

```bash
# From packages/cli
make build   # produces dist/skl (single-file executable)
./dist/skl --help
```

## Usage

```bash
# Show help
skl --help
skl run --help

# Train from a CSV and export ONNX (all flags, non-interactive)
skl run data/train.csv \
  --features x1,x2 \
  --label y \
  --task-type regression \
  --algorithm linear_regression \
  --format onnx \
  --output ./out/model.onnx \
  --no-interactive

# Fully interactive (prompts for every missing input)
skl run data/train.csv

# Clustering / anomaly detection (no label required)
skl run data/train.csv \
  --features x1,x2 \
  --task-type clustering \
  --algorithm kmeans \
  --format onnx \
  --no-interactive
```

### `skl run` flags

| Flag               | Purpose                                                          |
|--------------------|------------------------------------------------------------------|
| `csv_path` (arg)   | Path to the input CSV with a header row (required)               |
| `--output`, `-o`   | Output artefact path (default: `<csv stem>.<ext>`)               |
| `--task-type`, `-t`| Task type: `classification`, `regression`, `semi_supervised`, `clustering`, `anomaly_detection` |
| `--features`, `-f` | Comma-separated feature columns                                  |
| `--label`, `-l`    | Label column (required for supervised / semi-supervised tasks)   |
| `--algorithm`, `-a`| Algorithm id (prompted if omitted)                               |
| `--format`         | Export format: `onnx`, `pkl`, `joblib` (default: `onnx`)         |
| `--no-interactive` | Fail instead of prompting for missing input                      |

When a required input is omitted and stdin is a TTY, `skl` prompts
interactively (`inquirer`). With `--no-interactive` or a non-TTY stdin, a
missing required input causes a non-zero exit.

## Environment

| Variable       | Default                 | Description                      |
|----------------|-------------------------|----------------------------------|

`skl` reads no environment variables. It is fully offline: the `run` command
never contacts a backend.

## Development

From `packages/cli`:

```bash
make lint    # ruff check src tests
make format  # ruff format + ruff check --fix
make test    # pytest tests -v --cov=src/skl --cov-report=term-missing
make build   # pyinstaller --onefile --name skl src/skl/pyinstaller_entry.py
```

### Layout

```
packages/cli/
├── pyproject.toml   # packaging, dev deps (pytest, pyinstaller, ruff)
├── skl.spec         # PyInstaller spec (kept for reference; CI/build use the entry point)
├── Makefile         # lint / test / format / build
├── scripts/
│   └── install.sh   # downloads the latest release binary
├── src/skl/
│   ├── __init__.py   # package entry, exposes main()
│   ├── cli.py        # click group: run (offline)
│   ├── helpers.py    # inquirer prompts, tabulate display, formatting
│   ├── local.py      # offline train+export pipeline (no backend)
│   ├── pipeline.py   # sklearn training pipelines per task type
│   ├── estimator.py  # estimator construction
│   ├── export.py     # ONNX / pickle / joblib export
│   ├── preprocessor.py  # sklearn ColumnTransformer preprocessor
│   ├── pyinstaller_entry.py  # frozen entry point (package context preserved)
│   └── catalogue/    # algorithm catalogue (mirrors the backend)
├── tests/            # pytest suite
└── examples/         # demo CSV and run script
```

**See also:** [`local-development.md`](./local-development.md),
[`environment-variables.md`](./environment-variables.md).
