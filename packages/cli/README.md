# skl-cli

scikit-learn-client CLI — train and export ONNX from a CSV.

## Quick Start

### Prerequisites

- Python 3.14.8

### Installation

#### Option 1: Prebuilt binary (recommended)

```bash
# Download the latest binary from GitHub Releases
curl -fsSL https://github.com/hieudoanm/scikit-learn-client/releases/download/latest/skl -o "$HOME/.local/bin/skl"
chmod +x "$HOME/.local/bin/skl"
```

Or use the installer script:

```bash
bash packages/cli/scripts/install.sh
```

Ensure `$HOME/.local/bin` is on your `PATH`.

#### Option 2: From source

```bash
# From packages/cli
uv tool install -e .
# or
pip install -e .
```

### Usage

```bash
# Show help
skl --help

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
```

`skl run` trains and exports entirely offline — no backend, no database, no
network required. Omitted inputs are prompted interactively; pass
`--no-interactive` to fail instead.

## Development

```bash
# From packages/cli
make lint    # ruff check src tests
make format  # ruff format + ruff check --fix
make test    # pytest tests -v --cov=src/skl --cov-report=term-missing
make build   # pyinstaller --onefile --name skl src/skl/pyinstaller_entry.py
```

## License

[MIT License](LICENSE)
