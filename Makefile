.PHONY: setup test test-fast check-env lint eval

# Privacy defaults for every target (see engine/privacy.py, ADR-0007)
export HF_HUB_DISABLE_TELEMETRY=1
export MLFLOW_DISABLE_TELEMETRY=true
export DO_NOT_TRACK=1
export GRADIO_ANALYTICS_ENABLED=False

setup:            ## install all dependencies (CPU-only torch)
	uv sync

test:             ## full suite, including tests that download models
	uv run pytest

test-fast:        ## offline unit tests only
	uv run pytest -m "not network and not slow"

check-env:        ## print hardware / package / smoke-test report
	uv run python scripts/check_env.py

lint:
	uv run ruff check .

eval:             ## full evaluation suite (implemented in Phase 4)
	@echo "eval/run_all.py arrives in Phase 4" && exit 1
