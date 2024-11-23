# LangChain DataHub Integration

This project demonstrates how to integrate LangChain/LangSmith workflows into DataHub's metadata platform without modifying the core DataHub code.

## Prerequisites

- Python 3.8+
- LangSmith API access
- (Optional) DataHub running locally or accessible endpoint

## Quick Start

1. **Set up your environment**

```bash
# Clone the repository
git clone <repository-url>
cd langchain-datahub-integration

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Configure environment variables**

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
# Required for LangSmith integration:
LANGSMITH_API_KEY=ls-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="default"

# Optional for DataHub integration:
DATAHUB_GMS_URL=http://localhost:8080
DATAHUB_TOKEN=your_datahub_personal_access_token
```

3. **Try the dry run first**

```bash
# Run LangSmith ingestion in dry-run mode
python src/dry_run_example.py

# This will show the metadata that would be sent to DataHub
```

4. **Run with LangSmith integration**

```bash
# Ingest LangSmith runs
python src/langsmith_ingestion.py
```

5. **(Optional) Start DataHub and run full integration**

```bash
# Start DataHub using Docker
datahub docker quickstart

# Wait for DataHub to be ready (usually takes a few minutes)
# Then run the full integration
python src/langchain_example.py
```

## Project Structure

```
project/
├── src/
│   ├── dry_run_emitter.py     # Emitter for testing without DataHub
│   ├── langsmith_ingestion.py # LangSmith integration
│   ├── langchain_example.py   # Full LangChain example
│   └── metadata_setup.py      # DataHub metadata setup
├── metadata/
│   └── types/                 # Custom type definitions
│       ├── llm_model.json
│       ├── llm_chain.json
│       └── llm_prompt.json
└── requirements.txt
```

## Using the Dry Run Mode

The dry run mode allows you to see what metadata would be sent to DataHub without actually sending it:

```python
from src.dry_run_emitter import DryRunEmitter
from src.langchain_example import LangChainMetadataEmitter

# Initialize with dry run emitter
emitter = LangChainMetadataEmitter(emitter=DryRunEmitter())

# Run your example
emitter.run_example()
```

## LangSmith Integration

The LangSmith integration allows you to:
1. Fetch runs from your LangSmith projects
2. Transform them into DataHub metadata
3. Either preview (dry run) or ingest into DataHub

Example usage:
```python
from src.langsmith_ingestion import LangSmithIngestion
from src.dry_run_emitter import DryRunEmitter

# For dry run
ingestion = LangSmithIngestion(emitter=DryRunEmitter())
ingestion.ingest_recent_runs(limit=5)

# For actual ingestion
ingestion = LangSmithIngestion()
ingestion.ingest_recent_runs()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT
