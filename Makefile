.PHONY: install test lint clean dry-run run

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/
	black src/ tests/ --check
	isort src/ tests/ --check-only

format:
	black src/ tests/
	isort src/ tests/

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

dry-run:
	DATAHUB_DRY_RUN=true python src/langchain_example.py
	DATAHUB_DRY_RUN=true python src/langsmith_ingestion.py

run:
	python src/metadata_setup.py
	python src/langchain_example.py
	python src/langsmith_ingestion.py
