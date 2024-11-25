import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.emitters.datahub import DataHubEmitter
from src.emitters.console import ConsoleEmitter
from src.emitters.json_emitter import JSONEmitter
from src.base import LLMModel, LLMRun
from src.config import ObservabilityConfig
from datahub.metadata.schema_classes import MetadataChangeEventClass, DatasetSnapshotClass

@pytest.fixture
def sample_model():
    return LLMModel(
        name="test-model",
        provider="test-provider",
        model_family="test-family",
        capabilities=["test"],
        parameters={},
        metadata={}
    )

@pytest.fixture
def sample_run():
    return LLMRun(
        id="test-run",
        start_time=datetime.now(),
        end_time=datetime.now(),
        model=None,
        inputs={"test": "input"},
        outputs={"test": "output"},
        metrics={"latency": 1.0},
        parent_id=None,
        metadata={}
    )

def test_console_emitter(sample_model, sample_run, capsys):
    emitter = ConsoleEmitter()

    # Test model emission
    model_urn = emitter.emit_model(sample_model)
    captured = capsys.readouterr()
    assert "Model Metadata" in captured.out
    assert sample_model.name in captured.out

    # Test run emission
    run_urn = emitter.emit_run(sample_run)
    captured = capsys.readouterr()
    assert "Run Metadata" in captured.out
    assert sample_run.id in captured.out

def test_json_emitter(sample_model, sample_run, tmp_path):
    emitter = JSONEmitter(tmp_path)

    # Test model emission
    emitter.emit_model(sample_model)
    model_file = tmp_path / f"model_{sample_model.name}.json"
    assert model_file.exists()

    # Test run emission
    emitter.emit_run(sample_run)
    run_file = tmp_path / f"run_{sample_run.id}.json"
    assert run_file.exists()

def test_datahub_emitter(sample_model, sample_run):
    config = ObservabilityConfig()
    config.datahub_dry_run = True

    with patch('src.emitters.datahub.CustomDatahubRestEmitter') as mock_emitter:
        emitter = DataHubEmitter(debug=True)

        # Create proper MCE object
        mce = MetadataChangeEventClass(
            proposedSnapshot=DatasetSnapshotClass(
                urn="test:urn",
                aspects=[{"testAspect": {"key": "value"}}]
            )
        )

        # Test emission
        emitter.emit(mce)
        mock_emitter.return_value.emit.assert_called_once()
