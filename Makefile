.PHONY: install test lint clean dry-run run venv dev format check-connection setup-token docker-up docker-down docker-nuke flush-data flush-elastic flush-graph

PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin
DATAHUB_SECRET := $(shell cat .env | grep DATAHUB_SECRET | cut -d '=' -f2)
DATAHUB_TOKEN := $(shell cat .env | grep DATAHUB_TOKEN | cut -d '=' -f2)
DATAHUB_GMS_URL := $(shell cat .env | grep DATAHUB_GMS_URL | cut -d '=' -f2)

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
	@if [ ! -f .env ]; then \
		echo "Creating .env file..."; \
		cp .env.example .env; \
	fi
	@DATAHUB_SECRET=${DATAHUB_SECRET} datahub docker quickstart --quickstart-compose-file docker-compose-quickstart.yml

docker-down:
	datahub docker quickstart --stop

docker-nuke:
	datahub docker nuke

docker-sample:
	@DATAHUB_TOKEN=${DATAHUB_TOKEN} datahub docker ingest-sample-data --token ${DATAHUB_TOKEN}

check-connection:
	@echo "Testing DataHub connection..."
	curl -s -H "Authorization: Bearer ${DATAHUB_TOKEN}" ${DATAHUB_GMS_URL}/health | grep -q "true" \
		&& echo "Connection successful" \
		|| (echo "Connection failed. Make sure DataHub is running and your token is correct." && exit 1)

setup-token:
	@echo "Creating DataHub token..."
	@curl -X POST ${DATAHUB_GMS_URL}/api/v2/generate-personal-access-token \
		-H "Content-Type: application/json" \
		-d '{"type": "PERSONAL", "duration": "P30D", "name": "cli-token"}' \
		-u datahub:datahub \
		| jq -r '.accessToken' > .datahub-token
	@echo "Token saved to .datahub-token"
	@echo "Please add this token to your .env file as DATAHUB_TOKEN"

# Add these new targets
flush-data: flush-elastic flush-graph
	@echo "DataHub data has been flushed"

# Flush Elasticsearch indices
flush-elastic:
	@echo "Flushing Elasticsearch indices..."
	@curl -X POST ${DATAHUB_GMS_URL}/entities?action=batchDelete \
		-H "Authorization: Bearer ${DATAHUB_TOKEN}" \
		-H "Content-Type: application/json" \
		-d '{"urns": ["*"]}'

# Flush Neo4j graph
flush-graph:
	@echo "Flushing Neo4j graph..."
	@curl -X POST ${DATAHUB_GMS_URL}/graphql \
		-H "Authorization: Bearer ${DATAHUB_TOKEN}" \
		-H "Content-Type: application/json" \
		-d '{"query": "mutation { deleteGraph }"}'

# Optional: Add a safer version that only removes specific data
flush-langsmith-data:
	@echo "Flushing LangSmith data..."
	@curl -X POST ${DATAHUB_GMS_URL}/entities?action=batchDelete \
		-H "Authorization: Bearer ${DATAHUB_TOKEN}" \
		-H "Content-Type: application/json" \
		-d '{"urns": ["urn:li:dataset:(urn:li:dataPlatform:langsmith,*,PROD)"]}'
