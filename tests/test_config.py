import pytest
import os
from src.config import ObservabilityConfig, ObservabilitySetup

def test_config_from_env(monkeypatch):
    monkeypatch.setenv("DATAHUB_GMS_URL", "http://test:8080")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    config = ObservabilityConfig.from_env()
    assert config.datahub_gms_url == "http://test:8080"
    assert config.langsmith_api_key == "test-key"

def test_config_validation():
    config = ObservabilityConfig()
    config.datahub_dry_run = False
    config.datahub_gms_url = ""

    with pytest.raises(ValueError):
        config.validate()

def test_observability_setup():
    config = ObservabilityConfig()
    config.datahub_dry_run = True
    config.default_emitter = "console"

    setup = ObservabilitySetup(config)
    setup.setup()

    assert "console" in setup.emitters
    assert setup.get_emitter("console") is not None
