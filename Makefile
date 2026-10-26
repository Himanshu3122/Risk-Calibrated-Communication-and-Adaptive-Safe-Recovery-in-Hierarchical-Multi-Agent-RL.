PY ?= python

.PHONY: install dev lint test unit integration smoke smoke-pipeline clean

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

dev: install
	$(PY) -m pip install -r requirements-dev.txt

lint:
	$(PY) -m ruff check src scripts tests

test:
	$(PY) -m pytest

unit:
	$(PY) -m pytest tests/unit

integration:
	$(PY) -m pytest tests/integration

smoke:
	$(PY) -m pytest -m smoke

smoke-pipeline:
	$(PY) scripts/smoke_test.py --methods ricor h_mappo admac_h hpsf h_mappo_drop

clean:
	rm -rf runs .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +