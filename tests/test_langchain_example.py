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
    prompt = Mock(spec=ChatPromptTemplate)
    system_message = Mock(spec=SystemMessage)
    system_message.prompt.template = "You are a helpful assistant."
    system_message.__class__.__name__ = "SystemMessagePromptTemplate"

    human_message = Mock(spec=HumanMessage)
    human_message.prompt.template = "{question}"
    human_message.__class__.__name__ = "HumanMessagePromptTemplate"

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
    assert urn == f"urn:li:llmModel:{mock_llm.model_name}"
    assert emitter.is_dry_run == True


def test_emit_model_metadata(mock_llm):
    emitter = LangChainMetadataEmitter()
    emitter.emitter = DryRunEmitter()

    urn = emitter.emit_model_metadata(mock_llm)

    assert urn == f"urn:li:llmModel:{mock_llm.model_name}"
    emitted = emitter.emitter.get_emitted_mces()
    assert len(emitted) == 1
    model_props = emitted[0]["proposedSnapshot"]["aspects"][0]["llmModelProperties"]
    assert model_props["modelName"] == mock_llm.model_name
    assert model_props["modelType"] == "chat"


def test_emit_prompt_metadata(mock_prompt):
    emitter = LangChainMetadataEmitter()
    emitter.emitter = DryRunEmitter()

    urn = emitter.emit_prompt_metadata(mock_prompt)

    assert "urn:li:llmPrompt:" in urn
    emitted = emitter.emitter.get_emitted_mces()
    assert len(emitted) == 1
    prompt_props = emitted[0]["proposedSnapshot"]["aspects"][0]["llmPromptProperties"]
    assert prompt_props["templateFormat"] == "chat"
    assert isinstance(prompt_props["template"], str)


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
    assert chain_props["chainType"] == "RunnableSequence"
    assert chain_props["components"] == [model_urn, prompt_urn]


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
