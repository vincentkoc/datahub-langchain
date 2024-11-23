# LangChain DataHub Integration

This project demonstrates how to integrate LangChain/LangSmith workflows into DataHub's metadata platform, providing visibility into your LLM operations.

## Features

- Custom metadata types for LLM components:
  - Models (GPT-4, Claude, etc.)
  - Prompts (templates, few-shot examples)
  - Chains (LangChain sequences)
  - Runs (execution history from LangSmith)
- Dry run mode for testing without DataHub
- Automatic metadata ingestion from LangSmith
- Support for both local and remote DataHub instances

## Prerequisites

- Python 3.8+
- LangSmith API access
- DataHub instance (local or remote)
- OpenAI API key (for examples)

## Quick Start

1. **Set up environment**

```bash
# Clone and setup
git clone <repository-url>
cd langchain-datahub-integration
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy and edit environment variables
cp .env.example .env
```

2. **Configure environment**

Required variables in `.env`:
```bash
# LangSmith configuration
LANGSMITH_API_KEY=ls-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="default"

# OpenAI (for examples)
OPENAI_API_KEY=sk-...

# DataHub configuration
DATAHUB_GMS_URL=http://localhost:8080
DATAHUB_TOKEN=your_token_here
DATAHUB_DRY_RUN=false  # Set to true for testing
```

3. **Start DataHub (if running locally)**

```bash
# Start DataHub using provided compose file
make docker-up

# Wait for DataHub to be ready (~2-3 minutes)
# Then create a token and add it to .env
make setup-token
```

4. **Run the integration**

```bash
# Test with dry run first
make dry-run

# Run the full integration
make run
```

## Architecture

The integration consists of three main components:

1. **Metadata Setup** (`src/metadata_setup.py`)
   - Registers custom types with DataHub
   - Handles connection and authentication
   - Provides dry run capability

2. **LangSmith Integration** (`src/langsmith_ingestion.py`)
   - Fetches run history from LangSmith
   - Transforms runs into DataHub metadata
   - Handles batching and rate limiting

3. **LangChain Example** (`src/langchain_example.py`)
   - Shows live integration with LangChain
   - Demonstrates metadata emission
   - Includes error handling

## Custom Types

The integration defines several custom metadata types:

1. **LLM Model** (`metadata/types/llm_model.json`)
   - Model details (name, provider)
   - Capabilities and limitations
   - Performance metrics

2. **LLM Prompt** (`metadata/types/llm_prompt.json`)
   - Template structure
   - Input variables
   - Usage statistics

3. **LLM Chain** (`metadata/types/llm_chain.json`)
   - Component relationships
   - Configuration
   - Performance metrics

4. **LLM Run** (`metadata/types/llm_run.json`)
   - Execution details
   - Inputs/outputs
   - Error information

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
make test

# Format code
make format

# Run linters
make lint
```

## Troubleshooting

1. **DataHub Connection Issues**
   ```bash
   # Test DataHub connection
   make check-connection
   ```

2. **Dry Run Mode**
   ```bash
   # Set in .env
   DATAHUB_DRY_RUN=true

   # Or use make target
   make dry-run
   ```

3. **Common Issues**
   - Ensure DataHub is running (`docker ps`)
   - Check token is set in `.env`
   - Verify GMS URL is correct
   - Look for rate limiting in logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests (`make test`)
4. Submit a pull request

## License

GPL-3.0
