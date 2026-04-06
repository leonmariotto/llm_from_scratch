all: format pyright build pytest

build:
	uv build

pyright:
	uv run pyright

format:
	uv run ruff format

pytest:
	uv run pytest -vv

.PHONY: all ruff format pyright build pytest
