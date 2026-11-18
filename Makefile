.PHONY: setup test lint eval env
setup:
	uv sync
test:
	uv run pytest -q
lint:
	uv run ruff check .
eval:
	uv run python eval/run_all.py
env:
	uv run python scripts/env_report.py
