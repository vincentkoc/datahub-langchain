import pytest
from unittest.mock import Mock, patch
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
def test_datahub_emission_retry():
    with patch('src.emitters.datahub.CustomDatahubRestEmitter') as mock_emitter:
        # Setup mock to fail twice then succeed
        mock_emitter.return_value.emit.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            None  # Success on third try
        ]

        emitter = DataHubEmitter(debug=True)
        config = ObservabilityConfig()
        config.datahub_dry_run = False

        ingestor = LangsmithIngestor(config, emit_to_datahub=True)

        # Should succeed after retries
        run = Mock(id="retry-test-run", start_time=datetime.now())
        mce = ingestor._convert_run_to_mce(run)
        ingestor.emit_data([mce])

        assert mock_emitter.return_value.emit.call_count == 3
