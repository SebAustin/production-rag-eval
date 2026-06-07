.DEFAULT_GOAL := help
.PHONY: help sync download-data build-index calibrate ask serve eval eval-smoke ablation \
        lint type test ci fmt clean

UV ?= uv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sync:  ## Install dependencies (incl. dev) into the uv-managed venv
	$(UV) sync --all-extras

download-data:  ## Pull FinanceBench from HuggingFace into data/calibration/*.jsonl
	$(UV) run python scripts/download_financebench.py

build-index:  ## Contextualize + embed + index the calibration split
	$(UV) run python scripts/build_index.py

calibrate:  ## Fit the conformal abstention threshold on the calibration split
	$(UV) run python scripts/calibrate_abstention.py

ask:  ## Ask one question end-to-end. Usage: make ask Q="..."
	$(UV) run python -m rag_eval ask "$(Q)"

serve:  ## Run the FastAPI server (uvicorn, reload)
	$(UV) run uvicorn rag_eval.api.server:app --reload --port 8000

eval:  ## Full eval on the 30-question test split
	$(UV) run python -m evals.run_eval --limit 30 --seed 42

eval-smoke:  ## Smoke eval on 5 questions (fast)
	$(UV) run python -m evals.run_eval --limit 5 --seed 42

ablation:  ## Run the 5-configuration retriever ablation
	$(UV) run python scripts/run_ablation.py

lint:  ## ruff check + format check
	$(UV) run ruff check . && $(UV) run ruff format --check .

fmt:  ## ruff format (write)
	$(UV) run ruff format . && $(UV) run ruff check --fix .

type:  ## mypy --strict on src/
	$(UV) run mypy --strict src/

test:  ## pytest with coverage (mocked externals)
	$(UV) run pytest -q

ci: lint type test  ## Run the full local CI gate

clean:  ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov coverage.xml .coverage \
	       .eval_cache.sqlite .contextual_cache.sqlite
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
