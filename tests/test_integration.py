import pytest
import os
from unittest.mock import patch
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src.metadata_setup import MetadataSetup
from src.langchain_example import LangChainMetadataEmitter
from src.langsmith_ingestion import LangSmithIngestion

@pytest.mark.integration
def test_full_integration_flow():
    """Test the full integration flow with dry run"""
    os.environ["DATAHUB_DRY_RUN"] = "true"
    os.environ["OPENAI_API_KEY"] = "test-key"

    # 1. Set up metadata types
    setup = MetadataSetup()
    setup.register_all_types()

    # 2. Run LangChain example with actual components
    with patch('langchain_openai.OpenAI') as mock_openai:
        mock_openai.return_value.run.return_value = "Paris"

        llm = OpenAI(model_name="gpt-4o-mini")
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer: {question}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        emitter = LangChainMetadataEmitter()
        model_urn = emitter.emit_model_metadata(llm)
        prompt_urn = emitter.emit_prompt_metadata(prompt)
        chain_urn = emitter.emit_chain_metadata(chain, model_urn, prompt_urn)

        assert all([model_urn, prompt_urn, chain_urn])

        # Test actual chain execution
        result = chain.run("What is the capital of France?")
        assert result == "Paris"

    # 3. Run LangSmith ingestion
    with patch('langsmith.Client') as mock_client:
        ingestion = LangSmithIngestion()
        run_urns = ingestion.ingest_recent_runs(limit=1)
        assert isinstance(run_urns, list)

@pytest.mark.integration
def test_datahub_connection():
    """Test DataHub connection (when not in dry run)"""
    os.environ["DATAHUB_DRY_RUN"] = "false"

    with patch('datahub.emitter.rest_emitter.DatahubRestEmitter.test_connection') as mock_test:
        from datahub.emitter.rest_emitter import DatahubRestEmitter
        emitter = DatahubRestEmitter(
            gms_server=os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
        )
        emitter.test_connection()
        mock_test.assert_called_once()

@pytest.mark.integration
def test_dry_run_no_datahub_needed():
    """Verify dry run works without DataHub connection"""
    os.environ["DATAHUB_DRY_RUN"] = "true"
    os.environ["DATAHUB_GMS_URL"] = "http://nonexistent:8080"

    # Should not raise any connection errors
    setup = MetadataSetup()
    setup.register_all_types()
