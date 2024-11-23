import pytest
import os
from src.metadata_setup import MetadataSetup
from src.langchain_example import LangChainMetadataEmitter
from src.langsmith_ingestion import LangSmithIngestion

@pytest.mark.integration
def test_full_integration_flow():
    """Test the full integration flow with dry run"""
    os.environ["DATAHUB_DRY_RUN"] = "true"

    # 1. Set up metadata types
    setup = MetadataSetup()
    setup.register_all_types()

    # 2. Run LangChain example
    from src.langchain_example import run_example
    run_example()

    # 3. Run LangSmith ingestion
    from src.langsmith_ingestion import main
    main()

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
