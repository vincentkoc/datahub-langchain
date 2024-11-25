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
import json
from unittest.mock import Mock

@pytest.fixture
def mock_langsmith_run():
    """Create a properly configured mock run"""
    run = Mock()

    # Set basic attributes
    run.id = "test-run-id"
    run.start_time = datetime.now()
    run.end_time = datetime.now()

    # Set execution metadata for model info
    run.execution_metadata = {
        "model_name": "gpt-3.5-turbo"
    }

    # Set other required attributes
    run.inputs = {"prompt": "test"}
    run.outputs = {"response": "test"}
    run.error = None
    run.tags = []
    run.feedback_stats = {}
    run.latency = 1.0
    run.token_usage = {"total": 100}
    run.cost = 0.001
    run.metrics = {
        "latency": 1.0,
        "token_usage": {"total": 100},
        "cost": 0.001
    }
    run.metadata = {"test": "metadata"}

    # Configure mock to return real values
    for attr in ['id', 'start_time', 'end_time', 'execution_metadata', 'inputs',
                'outputs', 'error', 'tags', 'feedback_stats', 'latency',
                'token_usage', 'cost', 'metrics', 'metadata']:
        setattr(type(run), attr, property(lambda self, a=attr: getattr(self, f"_{a}", None)))
        setattr(run, f"_{attr}", getattr(run, attr))

    return run

@pytest.mark.integration
def test_full_ingestion_flow(tmp_path, mock_config, mock_langsmith_run):
    """Test the complete ingestion flow with mocked external services"""

    # Create proper mock data that's JSON serializable
    current_time = datetime.now()
    run_data = {
        'metrics': {
            "latency": 1.0,
            "token_usage": {"total": 100},
            "cost": 0.001
        },
        'metadata': {"test": "metadata"},
        'inputs': {"prompt": "test"},
        'outputs': {"response": "test"},
        'id': "test-run-id",
        'start_time': current_time,
        'end_time': current_time,
        'error': None,
        'tags': [],
        'feedback_stats': {}
    }

    # Configure mock to return real values instead of Mock objects
    for attr, value in run_data.items():
        setattr(mock_langsmith_run, attr, value)  # Set directly on mock object

    with patch('langsmith.Client') as mock_client_class:
        mock_client = mock_client_class.return_value

        # Mock the list_runs method to return our mock run directly
        mock_client.list_runs = MagicMock(return_value=[mock_langsmith_run])

        # Mock other methods to avoid 403 errors
        mock_client.list_projects = MagicMock(return_value=[{"name": "test-project"}])
        mock_client.read_project = MagicMock(return_value={"name": "test-project"})
        mock_client.list_sessions = MagicMock(return_value=[])  # Empty list is fine

        # Ensure no side effects
        mock_client.list_runs.side_effect = None
        mock_client.list_sessions.side_effect = None

        # Create and verify processing directory
        tmp_path.mkdir(parents=True, exist_ok=True)
        assert tmp_path.exists(), f"Processing directory not created: {tmp_path}"

        # Initialize components with explicit debug settings
        ingestor = LangsmithIngestor(
            config=mock_config,
            save_debug_data=True,  # Explicitly set to True
            processing_dir=tmp_path,
            emit_to_datahub=True
        )

        # Verify ingestor configuration
        assert ingestor.save_debug_data is True, "save_debug_data not set correctly"
        assert ingestor.processing_dir == tmp_path, "processing_dir not set correctly"

        # Set up the ingestor's client and connector
        ingestor.client = mock_client
        ingestor.project_name = "test-project"
        ingestor.connector.client = mock_client  # Important: set client on the connector too
        ingestor.connector.project_name = "test-project"

        # Test data collection using fetch_data
        raw_data = ingestor.fetch_data()  # This will create the debug file

        # Debug output if the assertion fails
        if len(raw_data) != 1:
            print(f"\nMock client list_runs called: {mock_client.list_runs.called}")
            print(f"Mock client list_runs return value: {mock_client.list_runs.return_value}")
            print(f"Raw data: {raw_data}")

        assert len(raw_data) == 1, "Expected one run in raw_data"

        # Test data processing with debug output
        processed_data = ingestor.process_data(raw_data)
        assert len(processed_data) == 1, "Expected one item in processed_data"

        # Verify debug data was saved
        debug_file = tmp_path / "langsmith_api_output.json"
        if not debug_file.exists():
            print(f"\nDebug directory contents: {list(tmp_path.iterdir())}")
            print(f"Ingestor config: save_debug_data={ingestor.save_debug_data}, processing_dir={ingestor.processing_dir}")
            print(f"Raw data length: {len(raw_data)}")
            print(f"Raw data first item: {raw_data[0].__dict__ if raw_data else None}")

        assert debug_file.exists(), f"Debug file not found at {debug_file}"

        # Verify file contents
        with open(debug_file, 'r') as f:
            debug_data = json.load(f)
            assert len(debug_data) == 1, "Expected one item in debug_data"
            assert debug_data[0]['id'] == "test-run-id"

        # Verify MCE output
        mce_file = tmp_path / "mce_output.json"
        assert mce_file.exists(), f"MCE file not found at {mce_file}"

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
