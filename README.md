# LangChain DataHub Integration 🔗

<p align="center">
  <strong>Seamless LLM Lineage for DataHub</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-GPL--3.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/LangChain-Integrated-orange.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/DataHub-Compatible-purple.svg" alt="DataHub">
  <br/>
  <img src="https://img.shields.io/github/stars/vincentkoc/natilius" alt="Stars">
  <img src="https://img.shields.io/github/forks/vincentkoc/natilius" alt="Forks">
  <img src="https://img.shields.io/github/issues/vincentkoc/natilius" alt="Issues">
</p>

A comprehensive observability solution that integrates LangChain and LangSmith workflows into DataHub's metadata platform, providing deep visibility into your LLM operations.

## Features

- 🔄 **Real-Time Observation**: Live monitoring of LangChain operations
- 📊 **Rich Metadata**: Detailed tracking of models, prompts, and chains
- 🔍 **Deep Insights**: Comprehensive metrics and lineage tracking
- 🚀 **Multiple Platforms**: Support for LangChain, LangSmith, and more
- 🛠 **Extensible**: Easy to add new platforms and emitters
- 🧪 **Debug Mode**: Built-in debugging and dry run capabilities

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd langchain-datahub-integration

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
```

## Quick Start

1. **Configure Environment**

```bash
# Required environment variables
LANGSMITH_API_KEY=ls-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=default

OPENAI_API_KEY=sk-...

DATAHUB_GMS_URL=http://localhost:8080
DATAHUB_TOKEN=your_token_here
```

2. **Run Basic Example**

```python
from langchain_openai import ChatOpenAI
from src.platforms.langchain import LangChainObserver
from src.emitters.datahub import DataHubEmitter
from src.config import ObservabilityConfig

# Setup observation
config = ObservabilityConfig(langchain_verbose=True)
emitter = DataHubEmitter(gms_server="http://localhost:8080")
observer = LangChainObserver(config=config, emitter=emitter)

# Initialize LLM with observer
llm = ChatOpenAI(callbacks=[observer])

# Run with automatic observation
response = llm.invoke("Tell me a joke")
```

## Architecture

The integration consists of three main components:

1. **Observers** (`src/platforms/`)
   - Real-time monitoring of LLM operations
   - Metric collection and event tracking
   - Platform-specific adapters

2. **Emitters** (`src/emitters/`)
   - DataHub metadata emission
   - Console debugging output
   - JSON file export

3. **Collectors** (`src/collectors/`)
   - Historical data collection
   - Batch processing
   - Aggregated metrics

## Usage Examples

### Basic LangChain Integration

```python
# examples/langchain_basic.py
from langchain_openai import ChatOpenAI
from src.platforms.langchain import LangChainObserver

observer = LangChainObserver(config=config, emitter=emitter)
llm = ChatOpenAI(callbacks=[observer])
```

### RAG Pipeline Integration

```python
# examples/langchain_rag.py
from langchain.chains import RetrievalQA
from src.utils.metrics import MetricsAggregator

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    callbacks=[observer]
)
```

### Historical Data Ingestion

```python
# examples/langsmith_ingest.py
from src.cli.ingest import ingest_logic

ingest_logic(
    days=7,
    platform='langsmith',
    debug=True,
    save_debug_data=True
)
```

## Customization

The integration is highly customizable through:

- **Configuration** (`src/config.py`): Environment and platform settings
- **Custom Emitters**: Implement `LLMMetadataEmitter` for new destinations
- **Platform Extensions**: Add new platforms by implementing `LLMPlatformConnector`
- **Metrics Collection**: Extend `MetricsAggregator` for custom metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests and linting:
   ```bash
   make test
   make lint
   ```
4. Submit a pull request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/vincentkoc">Vincent Koc</a>
</p>
