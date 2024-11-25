import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.emitters.datahub import DataHubEmitter, CustomDatahubRestEmitter
from src.base import LLMModel, LLMRun
from datahub.metadata.schema_classes import MetadataChangeEventClass

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
            Exception("First failure"),
            Exception("Second failure"),
            Mock(status_code=200)
        ]

        mce = MetadataChangeEventClass()
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
        assert mock_run.id in urn

def test_datahub_emitter_error_handling():
    """Test error handling in DataHub emitter"""
    emitter = DataHubEmitter(debug=True, hard_fail=False)

    # Test with invalid model
    invalid_model = Mock(spec=LLMModel)
    urn = emitter.emit_model(invalid_model)
    assert urn == ""

    # Test with invalid run
    invalid_run = Mock(spec=LLMRun)
    urn = emitter.emit_run(invalid_run)
    assert urn == ""

def test_datahub_platform_registration():
    """Test platform registration"""
    emitter = DataHubEmitter(debug=True)

    with patch.object(emitter.platform_extender, 'register_all_platforms') as mock_register:
        emitter.register_platforms()
        assert mock_register.called
