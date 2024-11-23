from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.langsmith_ingestion import LangSmithIngestion
from src.metadata_setup import DryRunEmitter


@pytest.fixture
def mock_run():
    run = Mock()
    run.id = "test-run-id"
    run.start_time = datetime.now()
    run.end_time = datetime.now()
    run.status = "completed"
    run.inputs = {"test": "input"}
    run.outputs = {"test": "output"}

    # Create proper execution metadata
    token_usage = {"prompt_tokens": 10, "completion_tokens": 20}
    run.execution_metadata = {"token_usage": token_usage}

    # Create empty feedback list
    run.feedback_list = []
    run.error = None
    run.runtime_seconds = 1.0
    run.parent_run_id = None
    run.child_run_ids = []
    run.tags = []

    # Mock the get method
    run.execution_metadata.get = lambda x, default=None: token_usage if x == "token_usage" else default
    return run


@pytest.fixture
def mock_client():
    with patch("langsmith.Client") as mock:
        client = Mock()
        mock.return_value = client
        yield client


def test_emit_run_metadata(mock_run):
    ingestion = LangSmithIngestion()
    ingestion.emitter = DryRunEmitter()

    urn = ingestion.emit_run_metadata(mock_run)

    assert urn == f"urn:li:llmRun:{mock_run.id}"
    emitted = ingestion.emitter.get_emitted_mces()
    assert len(emitted) == 1
    run_props = emitted[0]["proposedSnapshot"]["aspects"][0]["llmRunProperties"]
    assert run_props["runId"] == mock_run.id
    assert run_props["status"] == mock_run.status


def test_ingest_recent_runs(mock_client, mock_run):
    mock_client.list_runs.return_value = [mock_run]

    ingestion = LangSmithIngestion()
    ingestion.emitter = DryRunEmitter()
    ingestion.client = mock_client

    run_urns = ingestion.ingest_recent_runs(limit=1)

    assert len(run_urns) == 1
    assert run_urns[0] == f"urn:li:llmRun:{mock_run.id}"
    mock_client.list_runs.assert_called_once_with(limit=1)
