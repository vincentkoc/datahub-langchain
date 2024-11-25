import pytest
from datetime import datetime
from unittest.mock import Mock
from src.config import ObservabilityConfig

@pytest.fixture
def mock_config():
    """Mock ObservabilityConfig for testing"""
    config = ObservabilityConfig()
    config.datahub_dry_run = True
    config.langsmith_api_key = "test-key"
    config.default_emitter = "console"  # Use console emitter by default in tests
    return config

@pytest.fixture
def mock_langsmith_run():
    """Mock LangSmith run with proper dictionary returns"""
    run = Mock()

    # Create proper dictionaries for all attributes
    run_data = {
        'id': "test-run-id",
        'start_time': datetime.now(),
        'end_time': datetime.now(),
        'metrics': {"latency": 1.0, "cost": 0.01},
        'metadata': {"source": "test"},
        'inputs': {"prompt": "test prompt"},
        'outputs': {"response": "test response"},
        'execution_metadata': {
            "model_name": "gpt-4",
            "context_window": 8192,
            "max_tokens": 1000
        },
        'error': None,
        'tags': [],
        'feedback_stats': {},
        'token_usage': {"prompt_tokens": 10, "completion_tokens": 20}
    }

    # Set all attributes directly
    for attr, value in run_data.items():
        setattr(run, attr, value)

    return run

@pytest.fixture
def mock_client(mock_langsmith_run):
    """Mock LangSmith client with proper session handling"""
    client = Mock()

    # Create mock session
    mock_session = Mock()
    mock_session.id = "test-session"
    mock_session.name = "default"

    # Configure client with proper session/project flow
    client.list_sessions.return_value = [mock_session]
    client.read_session.return_value = mock_session
    client.list_runs.return_value = [mock_langsmith_run]
    client.list_projects.return_value = [{"name": "test-project"}]
    client.read_project.return_value = {"name": "test-project"}

    # Ensure no side effects
    client.list_runs.side_effect = None
    client.list_sessions.side_effect = None

    return client
