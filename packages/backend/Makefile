.PHONY: help lint format test coverage dev install clean all

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3.14
UV := uv

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(UV) sync --extra onnx

lint: ## Run linter and auto-fix issues
	$(UV) run ruff check src/ --fix

format: ## Format code with ruff
	$(UV) run ruff format src/

check: lint format ## Run all checks (lint + format)

test: ## Run unit tests
	$(UV) run pytest tests/api tests/db tests/export tests/storage tests/training tests/conftest.py -v

test-e2e: ## Run e2e tests (requires running server)
	$(UV) run pytest tests/test_e2e.py -v

test-all: test test-e2e ## Run all tests

coverage: ## Run tests with coverage report
	$(UV) run pytest tests/api tests/db tests/export tests/storage tests/training tests/conftest.py --cov=src --cov-report=html --cov-report=term

coverage-html: coverage ## Generate HTML coverage report
	@echo 'Coverage report generated in htmlcov/index.html'

dev: ## Start development server
	DATABASE_URL=postgresql+asyncpg://skl:skl@localhost:5432/skl \
	SKL_DATA_DIR=/tmp/skl-data \
	$(UV) run --extra onnx python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

clean: ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

all: install check test ## Install, check, and test
