.PHONY: lint typecheck check

lint:
	PYTHONPATH=src .venv/bin/ruff check src tests

typecheck:
	PYTHONPATH=src .venv/bin/mypy src

check: lint typecheck
