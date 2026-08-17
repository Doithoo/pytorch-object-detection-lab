.PHONY: lint format-check typecheck test build

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

build:
	uv run python -m build
