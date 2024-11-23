from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from freezegun import freeze_time

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

    # Create proper execution metadata with Mock
    token_usage = {"prompt_tokens": 10, "completion_tokens": 20}
    metadata = Mock()
    metadata.get = Mock(return_value=token_usage)
    run.execution_metadata = metadata

    # Create empty lists for required attributes
    run.child_run_ids = []  # Empty list instead of Mock
    run.feedback_list = []
    run.tags = []
    run.error = None
    run.runtime_seconds = 1.0
    run.parent_run_id = None
    run.latency = 0.5
    run.cost = 0.001
    run.name = "test_run"
    return run


@pytest.fixture
def mock_client():
    with patch("langsmith.Client") as mock:
        client = Mock()
        mock.return_value = client
        yield client


@freeze_time("2024-01-01")
def test_emit_run_metadata(mock_run):
    ingestion = LangSmithIngestion()
    ingestion.emitter = DryRunEmitter()

    urn = ingestion.emit_run_metadata(mock_run)
    expected_urn = f"urn:li:dataset:(urn:li:dataPlatform:llm,run_{mock_run.id},PROD)"
    assert urn == expected_urn

    emitted = ingestion.emitter.get_emitted_mces()
    assert len(emitted) == 1
    run_props = emitted[0]["proposedSnapshot"]["aspects"][0]["DatasetProperties"]
    assert run_props["customProperties"]["runId"] == mock_run.id
    assert run_props["customProperties"]["status"] == mock_run.status


@freeze_time("2024-01-01")
def test_ingest_recent_runs(mock_client, mock_run):
    mock_client.list_runs.return_value = [mock_run]

    ingestion = LangSmithIngestion()
    ingestion.emitter = DryRunEmitter()
    ingestion.client = mock_client

    run_urns = ingestion.ingest_recent_runs(limit=1)

    assert len(run_urns) == 1
    expected_urn = f"urn:li:dataset:(urn:li:dataPlatform:llm,run_{mock_run.id},PROD)"
    assert run_urns[0] == expected_urn

    # Get the actual call arguments
    call_args = mock_client.list_runs.call_args
    assert call_args is not None
    _, kwargs = call_args

    # Verify each argument individually
    assert kwargs["project_name"] == ingestion.project_name
    assert kwargs["execution_order"] == 1
    assert kwargs["limit"] == 1
    # Verify start_time is within expected range
    expected_start = datetime.now() - timedelta(days=7)
    actual_start = kwargs["start_time"]
    assert isinstance(actual_start, datetime)
    assert abs((actual_start - expected_start).total_seconds()) < 1  # Within 1 second
