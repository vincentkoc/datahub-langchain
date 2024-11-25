import pytest
from datetime import datetime
from src.base import LLMModel, LLMRun, LLMChain

def test_llm_model():
    model = LLMModel(
        name="test-model",
        provider="test-provider",
        model_family="test-family",
        capabilities=["test-capability"],
        parameters={"param": "value"},
        metadata={"meta": "data"}
    )

    assert model.name == "test-model"
    assert model.provider == "test-provider"
    assert "test-capability" in model.capabilities

def test_llm_run():
    run = LLMRun(
        id="test-run",
        start_time=datetime.now(),
        end_time=None,
        model=None,
        inputs={"input": "test"},
        outputs={},
        metrics={},
        parent_id=None,
        metadata={}
    )

    assert run.id == "test-run"
    assert run.inputs["input"] == "test"

def test_llm_chain():
    chain = LLMChain(
        id="test-chain",
        name="TestChain",
        components=["comp1", "comp2"],
        config={"key": "value"},
        metadata={}
    )

    assert chain.id == "test-chain"
    assert len(chain.components) == 2
