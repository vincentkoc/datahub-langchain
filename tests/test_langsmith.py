import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.platforms.langsmith import LangSmithConnector, LangsmithIngestor
from src.base import LLMModel, LLMRun
from src.config import ObservabilityConfig
from datahub.metadata.schema_classes import MetadataChangeEventClass

@pytest.fixture
def mock_langsmith_run():
    run = Mock()
    run.id = "test-run-id"
    run.start_time = datetime.now()
    run.end_time = datetime.now()
    run.execution_metadata = {
        "model_name": "gpt-4",
        "context_window": 8192,
        "max_tokens": 1000
    }
    run.inputs = {"prompt": "test prompt"}
    run.outputs = {"response": "test response"}
    run.error = None
    run.tags = []
    run.feedback_stats = {}
    run.latency = 1.0
    run.cost = 0.01
    run.token_usage = {"prompt_tokens": 10, "completion_tokens": 20}
    return run

@pytest.fixture
def mock_config():
    config = ObservabilityConfig()
    config.datahub_dry_run = True
    config.langsmith_api_key = "test-key"
    return config

def test_langsmith_connector(mock_config, mock_langsmith_run):
    with patch('langsmith.Client') as MockClient:
        # Configure mock client
        mock_client = MockClient.return_value
        mock_client.list_runs.return_value = [mock_langsmith_run]

        # Configure mock run attributes
        mock_langsmith_run.configure_mock(**{
            'metrics': {},
            'metadata': {},
            'inputs': {},
            'outputs': {},
            'execution_metadata': {'model_name': 'test-model'},
            'error': None,
            'tags': [],
            'feedback_stats': {}
        })

        connector = LangSmithConnector(mock_config)
        # Mock the project name to avoid API call
        connector.project_name = "test-project"

        runs = connector.get_runs()
        assert len(runs) == 1
        assert runs[0].id == mock_langsmith_run.id

def test_langsmith_ingestor(mock_config, mock_langsmith_run, tmp_path):
    with patch('langsmith.Client') as MockClient:
        # Configure mock client
        mock_client = MockClient.return_value
        mock_client.list_runs.return_value = [mock_langsmith_run]

        # Configure mock run with proper return values
        mock_langsmith_run.configure_mock(**{
            'metrics': {},
            'metadata': {},
            'inputs': {},
            'outputs': {},
            'execution_metadata': {'model_name': 'test-model'},
            'error': None,
            'tags': [],
            'feedback_stats': {}
        })

        ingestor = LangsmithIngestor(
            mock_config,
            save_debug_data=True,
            processing_dir=tmp_path,
            emit_to_datahub=False
        )

        # Mock the project name to avoid API call
        ingestor.project_name = "test-project"

        raw_data = ingestor.fetch_data()
        assert len(raw_data) == 1

        processed_data = ingestor.process_data(raw_data)
        assert len(processed_data) == 1
        assert isinstance(processed_data[0], MetadataChangeEventClass)

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
