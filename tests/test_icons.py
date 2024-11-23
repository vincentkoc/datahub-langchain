import pytest

from src.langchain_example import LangChainMetadataEmitter


@pytest.fixture
def sample_icons():
    """Sample base64 encoded icons for testing"""
    return {
        "langchain": "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiPjwvc3ZnPg==",
        "openai": "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiPjwvc3ZnPg=="
    }


def test_provider_icon_generation(sample_icons):
    emitter = LangChainMetadataEmitter()
    emitter.provider_icons = sample_icons  # Add icons to emitter

    # Test known provider
    openai_icon = emitter.get_provider_icon("OpenAI")
    assert openai_icon.startswith("data:image/svg+xml;base64,")

    # Test unknown provider
    unknown_icon = emitter.get_provider_icon("UnknownProvider")
    assert unknown_icon == ""


def test_chain_icon_generation(sample_icons):
    emitter = LangChainMetadataEmitter()

    # Test known chain type
    llm_chain_icon = emitter.get_chain_icon("LLMChain")
    assert llm_chain_icon.startswith("data:image/svg+xml;base64,")

    # Test fallback to LangChain icon
    unknown_chain_icon = emitter.get_chain_icon("UnknownChain")
    assert unknown_chain_icon.startswith("data:image/svg+xml;base64,")
    assert unknown_chain_icon == emitter.get_chain_icon("LangChain")
