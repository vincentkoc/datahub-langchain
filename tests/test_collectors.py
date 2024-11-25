import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
from src.collectors.run_collector import RunCollector
from src.collectors.model_collector import ModelCollector
from src.base import LLMModel, LLMRun

@pytest.fixture
def mock_connector():
    connector = Mock()
    connector.get_models.return_value = [
        LLMModel(
            name="test-model",
            provider="test-provider",
            model_family="test-family",
            capabilities=["test"],
            parameters={},
            metadata={}
        )
    ]
    connector.get_runs.return_value = [
        LLMRun(
            id="test-run",
            start_time=datetime.now(),
            end_time=datetime.now(),
            model=None,
            inputs={},
            outputs={},
            metrics={"latency": 1.0},
            parent_id=None,
            metadata={}
        )
    ]
    return connector

def test_run_collector(mock_connector):
    collector = RunCollector([mock_connector])
    runs = collector.collect_runs()

    assert len(runs) == 1
    assert runs[0].id == "test-run"

    stats = collector.get_run_stats(timedelta(days=1))
    assert "total_runs" in stats
    assert "average_latency" in stats

def test_model_collector(mock_connector):
    collector = ModelCollector([mock_connector])
    models = collector.collect_models()

    assert len(models) == 1
    assert models[0].name == "test-model"

    stats = collector.get_model_stats()
    assert "total_models" in stats
    assert "by_provider" in stats
