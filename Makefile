.PHONY: setup audit data coverage init-db features benchmarks models exploratory risk-validation freeze-evaluation test lint

PYTHON = PYTHONPATH=src LOKY_MAX_CPU_COUNT=1 MPLCONFIGDIR=.matplotlib-cache .venv/bin/python

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev]"

audit:
	$(PYTHON) -m eurostress.cli audit

data:
	$(PYTHON) -m eurostress.cli download

coverage:
	$(PYTHON) -m eurostress.cli coverage

init-db:
	$(PYTHON) -m eurostress.cli init-db

features:
	$(PYTHON) -m eurostress.cli build-features

benchmarks:
	$(PYTHON) -m eurostress.cli build-benchmarks

models:
	$(PYTHON) -m eurostress.cli fit-models

exploratory:
	$(PYTHON) -m eurostress.cli fit-exploratory

risk-validation:
	$(PYTHON) -m eurostress.cli risk-validation

freeze-evaluation:
	$(PYTHON) -m eurostress.cli freeze-evaluation

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .
