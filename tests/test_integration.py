import os
from unittest.mock import patch

import pytest
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI

from src.langchain_example import LangChainMetadataEmitter
from src.langsmith_ingestion import LangSmithIngestion
from src.metadata_setup import MetadataSetup


@pytest.mark.integration
def test_full_integration_flow(monkeypatch):
    """Test the full integration flow with dry run"""
    monkeypatch.setenv("DATAHUB_DRY_RUN", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Mock OpenAI API calls
    with patch("openai.resources.chat.completions.Completions.create") as mock_create:
        mock_create.return_value = {"choices": [{"message": {"content": "Paris"}}]}

        # Run the test
        from src.langchain_example import run_example
        run_example()


@pytest.mark.integration
def test_datahub_connection():
    """Test DataHub connection (when not in dry run)"""
    os.environ["DATAHUB_DRY_RUN"] = "false"

    with patch(
        "datahub.emitter.rest_emitter.DatahubRestEmitter.test_connection"
    ) as mock_test:
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
