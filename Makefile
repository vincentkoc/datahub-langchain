.PHONY: install test lint clean dry-run run venv dev format

PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

venv:
	$(PYTHON) -m venv $(VENV)
#	$(BIN)/python -m pip install --upgrade pip wheel setuptools

install: venv
	$(BIN)/pip install -r requirements.txt

dev: install
#	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

test: dev
	PYTHONPATH=. $(BIN)/pytest tests/ -v --cov=src --cov-report=term-missing

lint: dev
	$(BIN)/flake8 src/ tests/
	$(BIN)/black src/ tests/ --check
	$(BIN)/isort src/ tests/ --check-only
	$(BIN)/mypy src/ tests/

format: dev
	$(BIN)/black src/ tests/
	$(BIN)/isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf $(VENV)

dry-run:
	DATAHUB_DRY_RUN=true $(BIN)/python src/langchain_example.py
	DATAHUB_DRY_RUN=true $(BIN)/python src/langsmith_ingestion.py

run:
	$(BIN)/python src/metadata_setup.py
	$(BIN)/python src/langchain_example.py
	$(BIN)/python src/langsmith_ingestion.py

# Docker commands
docker-up:
	datahub docker quickstart

docker-down:
	datahub docker nuke
