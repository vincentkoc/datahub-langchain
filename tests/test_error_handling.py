import pytest
from src.langsmith_ingestion import LangSmithIngestion
from src.metadata_setup import DryRunEmitter, MetadataSetup
from src.langchain_example import LangChainMetadataEmitter

def test_failed_run_ingestion(mock_failed_run):
    """Test handling of failed LangSmith runs"""
    ingestion = LangSmithIngestion()
    ingestion.emitter = DryRunEmitter()

    urn = ingestion.emit_run_metadata(mock_failed_run)
    emitted = ingestion.emitter.get_emitted_mces()

    assert len(emitted) == 1
    assert emitted[0]["proposedSnapshot"]["aspects"][0]["llmRunProperties"]["status"] == "failed"
    assert emitted[0]["proposedSnapshot"]["aspects"][0]["llmRunProperties"]["error"] is not None

def test_datahub_emission_error(mock_datahub_error, mock_llm):
    """Test handling of DataHub emission errors"""
    emitter = LangChainMetadataEmitter()

    # Mock emitter to raise error
    class ErrorEmitter:
        def emit(self, _):
            raise mock_datahub_error

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
