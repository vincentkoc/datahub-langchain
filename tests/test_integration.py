import pytest
import os
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
    llm = OpenAI(model_name="text-davinci-003")
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

    # 3. Run LangSmith ingestion
    ingestion = LangSmithIngestion()
    run_urns = ingestion.ingest_recent_runs(limit=1)

    # In dry run mode, this should not raise errors
    assert isinstance(run_urns, list)

@pytest.mark.integration
def test_datahub_connection():
    """Test DataHub connection (when not in dry run)"""
    os.environ["DATAHUB_DRY_RUN"] = "false"

    from datahub.emitter.rest_emitter import DatahubRestEmitter
    emitter = DatahubRestEmitter(
        gms_server=os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
    )

    # Test connection - should raise exception if fails
    emitter.test_connection()
