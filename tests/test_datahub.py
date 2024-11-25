import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.emitters.datahub import DataHubEmitter, CustomDatahubRestEmitter
from src.base import LLMModel, LLMRun
from datahub.metadata.schema_classes import MetadataChangeEventClass, MLModelSnapshotClass, MLModelPropertiesClass

@pytest.fixture
def mock_model():
    return LLMModel(
        name="test-model",
        provider="OpenAI",
        model_family="GPT-3.5",
        capabilities=["text-generation"],
        parameters={"temperature": 0.7},
        metadata={"source": "test"}
    )

@pytest.fixture
def mock_run(mock_model):
    return LLMRun(
        id="test-run",
        start_time=datetime.now(),
        end_time=datetime.now(),
        model=mock_model,
        inputs={"prompt": "test"},
        outputs={"response": "test"},
        metrics={"latency": 1.0},
        parent_id=None,
        metadata={}
    )

def test_custom_emitter_initialization():
    """Test CustomDatahubRestEmitter initialization"""
    emitter = CustomDatahubRestEmitter(
        gms_server="http://test",
        token="test-token",
        debug=True
    )

    assert emitter._session.headers.get("Authorization") is not None
    assert emitter._session.timeout == 5

def test_custom_emitter_retry_logic():
    """Test retry logic in CustomDatahubRestEmitter"""
    emitter = CustomDatahubRestEmitter(gms_server="http://test")

    with patch.object(emitter._session, 'post') as mock_post:
        mock_post.side_effect = [
            Mock(status_code=500),  # First failure
            Mock(status_code=500),  # Second failure
            Mock(status_code=200)   # Success
        ]

        # Create a valid MCE for testing
        mce = MetadataChangeEventClass(
            proposedSnapshot=MLModelSnapshotClass(
                urn="urn:li:mlModel:(urn:li:dataPlatform:langchain,test,PROD)",
                aspects=[
                    MLModelPropertiesClass(
                        description="Test Model",
                        customProperties={"test": "value"}
                    )
                ]
            )
        )

        # Test retry logic
        emitter.emit(mce)
        assert mock_post.call_count == 3

def test_datahub_emitter_model_emission(mock_model):
    """Test model emission to DataHub"""
    emitter = DataHubEmitter(debug=True)

    with patch.object(emitter.emitter, 'emit') as mock_emit:
        urn = emitter.emit_model(mock_model)
        assert mock_emit.called
        assert isinstance(urn, str)
        assert "mlModel" in urn

def test_datahub_emitter_run_emission(mock_run):
    """Test run emission to DataHub"""
    emitter = DataHubEmitter(debug=True)

    with patch.object(emitter.emitter, 'emit') as mock_emit:
        urn = emitter.emit_run(mock_run)
        assert mock_emit.called
        assert isinstance(urn, str)
        assert "urn:li:mlModel:(urn:li:dataPlatform:langchain" in urn

def test_datahub_emitter_error_handling():
    """Test error handling in DataHub emitter"""
    emitter = DataHubEmitter(debug=True, hard_fail=False)

    # Create invalid model with required attributes
    invalid_model = Mock(spec=LLMModel)
    invalid_model.name = "test"
    invalid_model.provider = "test"
    invalid_model.model_family = "test"
    invalid_model.capabilities = []
    invalid_model.parameters = {}
    invalid_model.metadata = {}

    # Mock the _emit_with_retry method instead of emit
    with patch.object(emitter, '_emit_with_retry') as mock_emit:
        mock_emit.side_effect = Exception("Test error")
        urn = emitter.emit_model(invalid_model)
        assert urn == ""  # Should return empty string on error

    # Create invalid run with required attributes
    invalid_run = Mock(spec=LLMRun)
    invalid_run.id = "test"
    invalid_run.start_time = datetime.now()
    invalid_run.end_time = datetime.now()
    invalid_run.metrics = {}
    invalid_run.inputs = {}
    invalid_run.outputs = {}
    invalid_run.metadata = {}

    # Mock the _emit_with_retry method again
    with patch.object(emitter, '_emit_with_retry') as mock_emit:
        mock_emit.side_effect = Exception("Test error")
        urn = emitter.emit_run(invalid_run)
        assert urn == ""  # Should return empty string on error

def test_datahub_platform_registration():
    """Test platform registration"""
    emitter = DataHubEmitter(debug=True)

    with patch.object(emitter.platform_extender, 'register_all_platforms') as mock_register:
        emitter.register_platforms()
        assert mock_register.called
