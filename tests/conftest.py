import pytest
from datetime import datetime
from unittest.mock import Mock

@pytest.fixture
def sample_mce():
    """Sample Metadata Change Event for testing"""
    return {
        "proposedSnapshot": {
            "urn": "test:urn",
            "aspects": [{"testAspect": {"key": "value"}}]
        }
    }

@pytest.fixture
def sample_icons():
    """Sample base64 encoded icons for testing"""
    return {
        "langchain": "PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiPjwvc3ZnPg==",
        "openai": "PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiPjwvc3ZnPg=="
    }

@pytest.fixture
def mock_failed_run():
    """Mock LangSmith run with error"""
    run = Mock()
    run.id = "failed-run-id"
    run.start_time = datetime.now()
    run.end_time = datetime.now()
    run.status = "failed"
    run.error = "Test error message"
    run.inputs = {"test": "input"}
    run.outputs = None
    return run

@pytest.fixture
def mock_datahub_error():
    """Mock DataHub error response"""
    class DataHubError(Exception):
        pass
    return DataHubError("Failed to emit metadata")
