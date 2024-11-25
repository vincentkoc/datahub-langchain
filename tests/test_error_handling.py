import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.platforms.langsmith import LangsmithIngestor
from src.config import ObservabilityConfig
from src.emitters.datahub import DataHubEmitter

@pytest.fixture
def mock_failed_run():
    run = Mock()
    run.id = "failed-run-id"
    run.start_time = datetime.now()
    run.end_time = datetime.now()
    run.execution_metadata = {
        "model_name": "gpt-4",
        "error": "API timeout"
    }
    run.inputs = {"prompt": "test prompt"}
    run.outputs = None  # Failed run has no outputs
    run.error = "API timeout"
    run.tags = []
    run.feedback_stats = None
    run.latency = 0.0
    run.cost = 0.0
    run.token_usage = {}
    # Configure mock to return dictionary values
    run.metrics = {"error_rate": 1.0, "success_rate": 0.0}
    run.metadata = {"error": "API timeout"}
    return run

def test_failed_run_handling(mock_failed_run):
    config = ObservabilityConfig()
    config.datahub_dry_run = True

    ingestor = LangsmithIngestor(config)
    mce = ingestor._convert_run_to_mce(mock_failed_run)

    # Check that error information is properly captured
    props = mce.proposedSnapshot.aspects[0].customProperties
    assert props["outputs"] == "{}"  # Empty outputs for failed run
    assert "error" in props["metadata"]
    assert "API timeout" in props["metadata"]

def test_missing_metadata_handling():
    config = ObservabilityConfig()
    config.datahub_dry_run = True

    run = Mock()
    run.id = "incomplete-run-id"
    run.start_time = datetime.now()
    run.end_time = None
    run.execution_metadata = None  # Missing metadata
    run.inputs = {}
    run.outputs = {}
    run.error = None
    run.tags = []
    # Configure mock to return dictionary values instead of Mock objects
    run.metrics = {}
    run.metadata = {}
    run.configure_mock(**{
        'metrics': {},
        'metadata': {},
        'inputs': {},
        'outputs': {}
    })

    ingestor = LangsmithIngestor(config)
    mce = ingestor._convert_run_to_mce(run)

    # Check that missing metadata is handled gracefully
    props = mce.proposedSnapshot.aspects[0].customProperties
    assert props["end_time"] == ""
    assert props["metadata"] == "{}"

@pytest.mark.integration
def test_datahub_emission_retry(mock_config):
    with patch('src.emitters.datahub.CustomDatahubRestEmitter') as mock_emitter_class:
        # Setup mock emitter with proper retry behavior
        mock_emitter = mock_emitter_class.return_value
        mock_emitter.emit.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            None  # Success on third try
        ]

        # Create and configure the DataHub emitter
        emitter = DataHubEmitter(debug=True)
        emitter.emitter = mock_emitter  # Directly set the mocked emitter

        # Configure the ingestor with the emitter
        mock_config.datahub_dry_run = False
        mock_config.default_emitter = "datahub"  # Important: set default emitter

        ingestor = LangsmithIngestor(mock_config, emit_to_datahub=True)
        ingestor.emitter = emitter  # Directly set the emitter

        # Create a properly configured mock run with real dictionaries
        run = Mock()
        run.id = "retry-test-run"
        run.start_time = datetime.now()
        run.end_time = datetime.now()

        # Create proper dictionaries for JSON serialization
        run_data = {
            'metrics': {"latency": 1.0},
            'metadata': {},
            'inputs': {"test": "input"},
            'outputs': {"test": "output"},
            'execution_metadata': {"model": "test"},
            'error': None,
            'tags': [],
            'feedback_stats': {}
        }

        # Configure mock to return real dictionaries
        for attr, value in run_data.items():
            setattr(run, attr, value)

        mce = ingestor._convert_run_to_mce(run)
        ingestor.emit_data([mce])

        # Verify the emit method was called exactly 3 times
        assert mock_emitter.emit.call_count == 3
