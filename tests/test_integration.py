import pytest
from unittest.mock import patch
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
def test_full_ingestion_flow(tmp_path):
    """Test the complete ingestion flow with mocked external services"""

    # Mock LangSmith API responses
    mock_runs = [
        {
            "id": "test-run-1",
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "execution_metadata": {"model_name": "gpt-4"},
            "inputs": {"prompt": "test"},
            "outputs": {"response": "test"},
            "error": None,
            "tags": [],
            "feedback_stats": {},
            "latency": 1.0,
            "cost": 0.01,
            "token_usage": {"total": 100}
        }
    ]

    with patch('langsmith.Client') as mock_client:
        mock_client.return_value.list_runs.return_value = mock_runs
        mock_client.return_value.project_name = "test-project"

        # Setup configuration
        config = ObservabilityConfig()
        config.datahub_dry_run = True
        config.langsmith_api_key = "test-key"

        # Initialize components
        connector = LangSmithConnector(config)
        connector.project_name = "test-project"  # Mock project name

        ingestor = LangsmithIngestor(
            config,
            save_debug_data=True,
            processing_dir=tmp_path,
            emit_to_datahub=True
        )
        ingestor.project_name = "test-project"  # Mock project name

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

    with patch('src.emitters.datahub.CustomDatahubRestEmitter') as mock_emitter:
        # Mock successful response after retry
        mock_emitter.return_value.emit.side_effect = [
            Exception("First failure"),
            type('Response', (), {'status_code': 200})()
        ]

        emitter = DataHubEmitter(debug=True)

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
        emitter.emit(mce)
        assert mock_emitter.return_value.emit.call_count == 2

@pytest.mark.integration
def test_batch_processing():
    """Test processing of large batches of runs"""
    from unittest.mock import Mock

    # Create mock runs with proper dictionary returns
    mock_runs = []
    for i in range(100):
        run = Mock()
        run.id = f"batch-run-{i}"
        run.start_time = datetime.now()
        run.end_time = datetime.now()
        run.execution_metadata = {}
        run.inputs = {}
        run.outputs = {}
        run.error = None
        run.tags = []
        run.feedback_stats = {}
        run.metrics = {}  # Return dictionary instead of Mock
        run.metadata = {}  # Return dictionary instead of Mock
        mock_runs.append(run)

    config = ObservabilityConfig()
    config.datahub_dry_run = True
    config.ingest_batch_size = 10

    ingestor = LangsmithIngestor(config)
    ingestor.project_name = "test-project"  # Mock project name
    processed_data = ingestor.process_data(mock_runs)
    assert len(processed_data) == 100
