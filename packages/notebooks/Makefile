.PHONY: lint format run export-datasets

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

run:
	uv run jupyter nbconvert --to notebook --execute pipeline.ipynb --output pipeline.ipynb

export-datasets:
	uv run jupyter nbconvert --to notebook --execute datasets.ipynb --output datasets.ipynb
