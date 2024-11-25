import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from src.platforms.langsmith import LangSmithConnector, LangsmithIngestor
from src.emitters.datahub import DataHubEmitter
from src.config import ObservabilityConfig
from datahub.metadata.schema_classes import (
    MetadataChangeEventClass,
    DatasetSnapshotClass,
    DatasetPropertiesClass
)

@pytest.mark.integration
def test_full_ingestion_flow(tmp_path, mock_config, mock_langsmith_run):
    """Test the complete ingestion flow with mocked external services"""

    with patch('langsmith.Client') as mock_client_class:
        # Create a fresh mock client for this test
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the session
        mock_session = MagicMock()
        mock_session.id = "test-session"
        mock_session.name = "default"

        # Configure client responses
        mock_client.list_sessions.return_value = [mock_session]
        mock_client.read_session.return_value = mock_session
        mock_client.list_runs.return_value = [mock_langsmith_run]
        mock_client.list_projects.return_value = [{"name": "test-project"}]
        mock_client.read_project.return_value = {"name": "test-project"}

        # Ensure no side effects
        mock_client.list_runs.side_effect = None
        mock_client.list_sessions.side_effect = None

        # Initialize components
        connector = LangSmithConnector(mock_config)
        connector.client = mock_client
        connector.project_name = "test-project"

        ingestor = LangsmithIngestor(
            mock_config,
            save_debug_data=True,
            processing_dir=tmp_path,
            emit_to_datahub=True
        )
        ingestor.client = mock_client
        ingestor.project_name = "test-project"

        # Test data collection
        runs = connector.get_runs(
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
        assert len(runs) == 1

        # Test data processing
        processed_data = ingestor.process_data(runs)
        assert len(processed_data) == 1

        # Verify debug data was saved
        assert (tmp_path / "langsmith_api_output.json").exists()
        assert (tmp_path / "mce_output.json").exists()

@pytest.mark.integration
def test_datahub_connection():
    """Test DataHub connectivity with retry logic"""
    config = ObservabilityConfig()
    config.datahub_dry_run = False
    config.default_emitter = "datahub"

    with patch('src.emitters.datahub.CustomDatahubRestEmitter') as mock_emitter_class:
        # Create a fresh mock emitter
        mock_emitter = MagicMock()
        mock_emitter_class.return_value = mock_emitter

        # Configure retry behavior
        mock_emitter.emit.side_effect = [
            Exception("First failure"),
            None  # Success on second try
        ]

        emitter = DataHubEmitter(debug=True)
        emitter.emitter = mock_emitter

        # Create proper MCE object
        mce = MetadataChangeEventClass(
            proposedSnapshot=DatasetSnapshotClass(
                urn="test:urn",
                aspects=[
                    DatasetPropertiesClass(
                        name="test",
                        description="test",
                        customProperties={"test": "value"}
                    )
                ]
            )
        )

        # Test emission with proper MCE object
        try:
            emitter.emit(mce)
        except Exception:
            pass  # First failure is expected
        emitter.emit(mce)  # Second attempt should succeed

        assert mock_emitter.emit.call_count == 2

@pytest.mark.integration
def test_batch_processing():
    """Test processing of large batches of runs"""
    # Create mock runs with proper dictionary returns
    mock_runs = []
    for i in range(100):
        run = MagicMock()
        run_data = {
            'id': f"batch-run-{i}",
            'start_time': datetime.now(),
            'end_time': datetime.now(),
            'metrics': {"latency": 1.0},
            'metadata': {},
            'inputs': {"test": "input"},
            'outputs': {"test": "output"},
            'execution_metadata': {"model": "test"},
            'error': None,
            'tags': [],
            'feedback_stats': {}
        }
        # Set all attributes directly
        for attr, value in run_data.items():
            setattr(run, attr, value)
        mock_runs.append(run)

    config = ObservabilityConfig()
    config.datahub_dry_run = True
    config.ingest_batch_size = 10

    ingestor = LangsmithIngestor(config)
    ingestor.project_name = "test-project"
    processed_data = ingestor.process_data(mock_runs)
    assert len(processed_data) == 100
