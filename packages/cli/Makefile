.PHONY: dev test lint format install clean build coverage

dev:
	@echo "Run: uv run skl --help"

test:
	@uv run python -m pytest tests -q

coverage:
	@uv run python -m pytest tests --cov=src --cov-report=term --cov-fail-under=80

lint:
	@uv run ruff check src tests

format:
	@uv run ruff format src tests
	@uv run ruff check --fix src tests

install:
	@if command -v uv >/dev/null 2>&1; then uv tool install -e .; else echo "uv not found; skipping cli install"; fi

build:
	@uv run pyinstaller src/skl/pyinstaller_entry.py --distpath $(CURDIR)/dist --workpath $(CURDIR)/build --noconfirm

clean:
	@rm -rf build dist *.egg-info src/*.egg-info .coverage htmlcov
