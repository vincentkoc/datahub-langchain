import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.langchain_example import LangChainMetadataEmitter
from src.langsmith_ingestion import LangSmithIngestion
from src.metadata_setup import DryRunEmitter, MetadataSetup


@pytest.fixture
def mock_failed_run():
    run = Mock()
    run.id = "failed-run-id"
    run.start_time = datetime.now()
    run.end_time = datetime.now()
    run.status = "failed"
    run.error = "Test error message"
    run.inputs = {"test": "input"}
    run.outputs = None

    # Create metadata with Mock
    token_usage = {"prompt_tokens": 10, "completion_tokens": 20}
    metadata = Mock()
    metadata.get = Mock(return_value=token_usage)
    run.execution_metadata = metadata

    # Add required attributes with proper numeric values
    run.child_run_ids = []  # Empty list instead of Mock
    run.feedback_list = []
    run.tags = []
    run.runtime_seconds = 1.0  # Use float instead of Mock
    run.parent_run_id = None
    run.latency = 0.5  # Use float instead of Mock
    run.cost = 0.001  # Use float instead of Mock
    run.name = "failed_test_run"  # Use string instead of Mock
    return run


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.model_name = "gpt-4o-mini"
    llm.model_kwargs = {"temperature": 0.7}
    return llm


def test_failed_run_ingestion(mock_failed_run):
    """Test handling of failed LangSmith runs"""
    ingestion = LangSmithIngestion()
    ingestion.emitter = DryRunEmitter()

    urn = ingestion.emit_run_metadata(mock_failed_run)
    emitted = ingestion.emitter.get_emitted_mces()

    assert len(emitted) == 1
    run_props = emitted[0]["proposedSnapshot"]["aspects"][0]["DatasetProperties"]
    assert run_props["customProperties"]["status"] == "failed"
    assert run_props["customProperties"]["error"] == "Test error message"
    assert isinstance(run_props["customProperties"]["runtime"], float)
    assert isinstance(run_props["customProperties"]["metrics"]["latency"], float)
    assert isinstance(run_props["customProperties"]["metrics"]["cost"], float)


def test_datahub_emission_error(mock_datahub_error, mock_llm):
    """Test handling of DataHub emission errors"""
    emitter = LangChainMetadataEmitter()
    emitter.is_dry_run = False  # Force non-dry run mode

    # Mock emitter to raise error
    class ErrorEmitter:
        def emit(self, _):
            raise mock_datahub_error
        def test_connection(self):
            pass  # Don't raise error during connection test

    emitter.emitter = ErrorEmitter()

    with pytest.raises(Exception) as exc_info:
        emitter.emit_model_metadata(mock_llm)

    assert str(mock_datahub_error) in str(exc_info.value)


def test_invalid_type_registration(tmp_path):
    """Test handling of invalid type registration"""
    types_dir = tmp_path / "types"
    types_dir.mkdir()

    # Create invalid type file
    with open(types_dir / "invalid.json", "w") as f:
        f.write("invalid json")

    setup = MetadataSetup()
    setup.types_dir = types_dir
    setup.emitter = DryRunEmitter()

    # Should not raise exception but log error
    setup.register_all_types()
    assert len(setup.emitter.get_emitted_mces()) == 0
