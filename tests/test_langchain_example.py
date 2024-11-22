from unittest.mock import Mock, patch

import pytest
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI

from src.langchain_example import LangChainMetadataEmitter
from src.metadata_setup import DryRunEmitter


@pytest.fixture
def mock_llm():
    llm = Mock(spec=ChatOpenAI)
    llm.model_name = "gpt-4o-mini"
    llm.model_kwargs = {"temperature": 0.7}
    return llm


@pytest.fixture
def mock_prompt():
    system_message = Mock()
    system_message.type = "system"
    system_message.prompt = Mock()
    system_message.prompt.template = "You are a helpful assistant."
    system_message.__class__.__name__ = "SystemMessagePromptTemplate"

    human_message = Mock()
    human_message.type = "human"
    human_message.prompt = Mock()
    human_message.prompt.template = "{question}"
    human_message.__class__.__name__ = "HumanMessagePromptTemplate"

    prompt = Mock(spec=ChatPromptTemplate)
    prompt.messages = [system_message, human_message]
    prompt.input_variables = ["question"]
    return prompt


@pytest.fixture
def mock_chain():
    chain = Mock(spec=RunnableSequence)
    chain.__class__.__name__ = "RunnableSequence"
    return chain


def test_dry_run_mode(mock_llm, monkeypatch):
    """Test emitter behavior in dry run mode"""
    monkeypatch.setenv("DATAHUB_DRY_RUN", "true")
    emitter = LangChainMetadataEmitter()

    urn = emitter.emit_model_metadata(mock_llm)
    expected_urn = f"urn:li:dataset:(urn:li:dataPlatform:llm,model_{mock_llm.model_name},PROD)"
    assert urn == expected_urn
    assert emitter.is_dry_run == True


def test_emit_model_metadata(mock_llm):
    emitter = LangChainMetadataEmitter()
    emitter.emitter = DryRunEmitter()

    urn = emitter.emit_model_metadata(mock_llm)
    expected_urn = f"urn:li:dataset:(urn:li:dataPlatform:llm,model_{mock_llm.model_name},PROD)"
    assert urn == expected_urn

    emitted = emitter.emitter.get_emitted_mces()
    assert len(emitted) == 1
    model_props = emitted[0]["proposedSnapshot"]["aspects"][0]["DatasetProperties"]
    assert model_props["name"] == mock_llm.model_name
    assert model_props["customProperties"]["modelType"] == "chat"


def test_emit_prompt_metadata(mock_prompt):
    emitter = LangChainMetadataEmitter()
    emitter.emitter = DryRunEmitter()

    urn = emitter.emit_prompt_metadata(mock_prompt)
    assert "urn:li:dataset:(urn:li:dataPlatform:llm,prompt_" in urn
    assert ",PROD)" in urn

    emitted = emitter.emitter.get_emitted_mces()
    assert len(emitted) == 1
    prompt_props = emitted[0]["proposedSnapshot"]["aspects"][0]["DatasetProperties"]
    assert prompt_props["customProperties"]["templateFormat"] == "chat"
    assert isinstance(prompt_props["customProperties"]["template"], str)


def test_emit_chain_metadata(mock_chain, mock_llm, mock_prompt):
    emitter = LangChainMetadataEmitter()
    emitter.emitter = DryRunEmitter()

    model_urn = f"urn:li:dataset:(urn:li:dataPlatform:llm,model_{mock_llm.model_name},PROD)"
    prompt_urn = "urn:li:dataset:(urn:li:dataPlatform:llm,prompt_test,PROD)"

    urn = emitter.emit_chain_metadata(mock_chain, model_urn, prompt_urn)
    assert "urn:li:dataset:(urn:li:dataPlatform:llm,chain_" in urn
    assert ",PROD)" in urn

    emitted = emitter.emitter.get_emitted_mces()
    assert len(emitted) == 1
    chain_props = emitted[0]["proposedSnapshot"]["aspects"][0]["DatasetProperties"]
    assert chain_props["customProperties"]["chainType"] == "RunnableSequence"
    assert chain_props["customProperties"]["components"] == [model_urn, prompt_urn]


@pytest.mark.integration
def test_run_example_with_dry_run(monkeypatch):
    """Test the full example run in dry run mode"""
    monkeypatch.setenv("DATAHUB_DRY_RUN", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch("langchain_openai.ChatOpenAI") as mock_chat:
        mock_chat.return_value.invoke.return_value.content = "Paris"
        from src.langchain_example import run_example

        # Should not raise any exceptions in dry run mode
        run_example()
