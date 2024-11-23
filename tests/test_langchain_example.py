import pytest
from unittest.mock import Mock, patch
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src.langchain_example import LangChainMetadataEmitter
from src.metadata_setup import DryRunEmitter

@pytest.fixture
def mock_llm():
    llm = Mock(spec=OpenAI)
    llm.model_name = "test-model"
    llm.model_kwargs = {"temperature": 0.7}
    return llm

@pytest.fixture
def mock_prompt():
    prompt = Mock(spec=PromptTemplate)
    prompt.template = "Test template"
    prompt.input_variables = ["test_var"]
    return prompt

@pytest.fixture
def mock_chain():
    chain = Mock(spec=LLMChain)
    chain.__class__.__name__ = "LLMChain"
    chain.verbose = True
    chain.max_retries = 2
    chain.callbacks = []
    return chain

def test_dry_run_mode(mock_llm, monkeypatch):
    """Test emitter behavior in dry run mode"""
    monkeypatch.setenv("DATAHUB_DRY_RUN", "true")
    emitter = LangChainMetadataEmitter()

    urn = emitter.emit_model_metadata(mock_llm)
    assert urn == f"urn:li:llmModel:{mock_llm.model_name}"
    assert emitter.is_dry_run == True

def test_emit_model_metadata(mock_llm):
    emitter = LangChainMetadataEmitter()
    emitter.emitter = DryRunEmitter()

    urn = emitter.emit_model_metadata(mock_llm)

    assert urn == f"urn:li:llmModel:{mock_llm.model_name}"
    emitted = emitter.emitter.get_emitted_mces()
    assert len(emitted) == 1
    assert emitted[0]["proposedSnapshot"]["aspects"][0]["llmModelProperties"]["modelName"] == mock_llm.model_name

def test_emit_prompt_metadata(mock_prompt):
    emitter = LangChainMetadataEmitter()
    emitter.emitter = DryRunEmitter()

    urn = emitter.emit_prompt_metadata(mock_prompt)

    assert "urn:li:llmPrompt:" in urn
    emitted = emitter.emitter.get_emitted_mces()
    assert len(emitted) == 1
    assert emitted[0]["proposedSnapshot"]["aspects"][0]["llmPromptProperties"]["template"] == mock_prompt.template

def test_emit_chain_metadata(mock_chain, mock_llm, mock_prompt):
    emitter = LangChainMetadataEmitter()
    emitter.emitter = DryRunEmitter()

    model_urn = f"urn:li:llmModel:{mock_llm.model_name}"
    prompt_urn = "urn:li:llmPrompt:test"

    urn = emitter.emit_chain_metadata(mock_chain, model_urn, prompt_urn)

    assert "urn:li:llmChain:" in urn
    emitted = emitter.emitter.get_emitted_mces()
    assert len(emitted) == 1
    chain_props = emitted[0]["proposedSnapshot"]["aspects"][0]["llmChainProperties"]
    assert chain_props["chainType"] == "LLMChain"
    assert chain_props["components"] == [model_urn, prompt_urn]

@pytest.mark.integration
def test_run_example_with_dry_run(monkeypatch):
    """Test the full example run in dry run mode"""
    monkeypatch.setenv("DATAHUB_DRY_RUN", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch('langchain_openai.OpenAI') as mock_openai:
        mock_openai.return_value.run.return_value = "Paris"
        from src.langchain_example import run_example

        # Should not raise any exceptions in dry run mode
        run_example()
