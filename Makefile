all: format ruff pyright build pytest

ruff:
	uv run ruff check

build:
	uv build

pyright:
	uv run pyright

format:
	uv run ruff format --check

pytest:
	uv run pytest -vv

.PHONY: all ruff format pyright build pytest
