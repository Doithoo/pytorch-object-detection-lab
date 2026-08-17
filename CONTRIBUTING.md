# Contributing

Install Python 3.10-3.12 and uv, then run `uv sync --locked --extra dev`. Keep changes focused and add a failing test before production behavior.

Before submitting, run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and `uv run pytest -W error::DeprecationWarning`. Tests must be offline: never add a test that downloads datasets, model weights, fonts, or other assets. Use synthetic VOC fixtures and fake detectors.

Use English ASCII Conventional Commits such as `feat(evaluation): Add report export`. Keep the subject imperative, at most 72 characters, and explain what and why in the body. New datasets implement the manifest-backed dataset boundary; new models register an explicit constructor and weight policy. Update matching English and Chinese documentation.
