import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from datahub.metadata.schema_classes import (
    MLModelSnapshotClass,
    MLModelPropertiesClass,
    MetadataChangeEventClass
)

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

    # Configure mock to return dictionary values
    run_data = {
        'inputs': {"prompt": "test prompt"},
        'outputs': None,  # Failed run has no outputs
        'error': "API timeout",
        'tags': [],
        'feedback_stats': None,
        'latency': 0.0,
        'cost': 0.0,
        'token_usage': {},
        'metrics': {"error_rate": 1.0, "success_rate": 0.0},
        'metadata': {"error": "API timeout"}
    }

    # Configure mock to return real dictionaries
    for attr, value in run_data.items():
        setattr(type(run), attr, property(lambda self, v=value: v))

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

    # Configure mock with minimal data
    run_data = {
        'inputs': {},
        'outputs': {},
        'error': None,
        'tags': [],
        'metrics': {},
        'metadata': {}
    }

    # Configure mock to return real dictionaries
    for attr, value in run_data.items():
        setattr(type(run), attr, property(lambda self, v=value: v))

    ingestor = LangsmithIngestor(config)
    mce = ingestor._convert_run_to_mce(run)

    # Check that missing metadata is handled gracefully
    props = mce.proposedSnapshot.aspects[0].customProperties
    assert props["end_time"] == ""
    assert props["metadata"] == "{}"

@pytest.mark.integration
def test_datahub_emission_retry(mock_config):
    """Test retry mechanism for DataHub emissions"""
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

        # Force dry run to false to ensure emission
        emitter.config.datahub_dry_run = False

        # Create a test MCE
        mce = MetadataChangeEventClass(
            proposedSnapshot=MLModelSnapshotClass(
                urn="urn:li:mlModel:(test,test,PROD)",
                aspects=[
                    MLModelPropertiesClass(
                        description="Test Model",
                        customProperties={"test": "value"}
                    )
                ]
            )
        )

        # Test the retry mechanism directly
        emitter._emit_with_retry(mce)

        # Verify the emit method was called exactly 3 times
        assert mock_emitter.emit.call_count == 3
