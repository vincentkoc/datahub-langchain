import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.platforms.langsmith import LangSmithConnector, LangsmithIngestor
from src.base import LLMModel, LLMRun
from src.config import ObservabilityConfig
from datahub.metadata.schema_classes import MetadataChangeEventClass

@pytest.fixture
def mock_langsmith_run():
    # Create a mock with all required attributes as real dictionaries
    run = MagicMock()
    run.id = "test-run-id"
    run.start_time = datetime.now()
    run.end_time = datetime.now()
    run.execution_metadata = {
        "model_name": "gpt-4",
        "context_window": 8192,
        "max_tokens": 1000
    }

    # Configure mock to return dictionaries instead of Mock objects
    run.configure_mock(**{
        'metrics': {"latency": 1.0, "cost": 0.01},
        'metadata': {"source": "test"},
        'inputs': {"prompt": "test prompt"},
        'outputs': {"response": "test response"},
        'error': None,
        'tags': [],
        'feedback_stats': {},
        'token_usage': {"prompt_tokens": 10, "completion_tokens": 20}
    })

    # Ensure these return the configured values
    run.metrics = run.metrics
    run.metadata = run.metadata
    run.inputs = run.inputs
    run.outputs = run.outputs

    return run

@pytest.fixture
def mock_client(mock_langsmith_run):
    with patch('langsmith.Client') as MockClient:
        client = MockClient.return_value
        # Mock the list_runs method to return our mock run
        client.list_runs.return_value = [mock_langsmith_run]
        # Mock project name
        client.project_name = "test-project"
        yield client

def test_langsmith_connector(mock_config, mock_client, mock_langsmith_run):
    with patch('langsmith.Client') as MockClient:
        # Configure mock client
        mock_client = MockClient.return_value
        mock_client.list_runs.return_value = [mock_langsmith_run]
        mock_client.list_runs.side_effect = None  # Override any previous side_effect

        connector = LangSmithConnector(mock_config)
        connector.client = mock_client
        connector.project_name = "test-project"

        runs = connector.get_runs()
        assert len(runs) == 1
        assert runs[0].id == mock_langsmith_run.id

def test_langsmith_ingestor(mock_config, mock_client, mock_langsmith_run, tmp_path):
    # Configure mock run with proper data structure
    mock_langsmith_run.id = "test-run-id"
    mock_langsmith_run.start_time = datetime.now()
    mock_langsmith_run.end_time = datetime.now()
    mock_langsmith_run.error = None
    mock_langsmith_run.metrics = {
        "latency": 1.0,
        "token_usage": {"total": 100}
    }
    mock_langsmith_run.inputs = {"prompt": "test"}
    mock_langsmith_run.outputs = {"response": "test"}

    # Configure mock client
    mock_client.list_runs.return_value = [mock_langsmith_run]

    # Rest of the test...

def test_run_conversion(mock_config, mock_langsmith_run):
    ingestor = LangsmithIngestor(mock_config)
    mce = ingestor._convert_run_to_mce(mock_langsmith_run)

    assert isinstance(mce, MetadataChangeEventClass)
    assert "langsmith" in mce.proposedSnapshot.urn
    assert mock_langsmith_run.id in mce.proposedSnapshot.urn

    # Verify custom properties are properly formatted
    props = mce.proposedSnapshot.aspects[0].customProperties
    assert props["run_id"] == str(mock_langsmith_run.id)
    assert isinstance(props["inputs"], str)
    assert isinstance(props["outputs"], str)
