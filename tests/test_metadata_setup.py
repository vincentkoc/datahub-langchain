from pathlib import Path
import json

import pytest

from src.metadata_setup import (DryRunEmitter, MetadataSetup,
                                get_datahub_emitter)


def test_dry_run_emitter():
    emitter = DryRunEmitter()
    test_mce = {
        "proposedSnapshot": {
            "urn": "test:urn",
            "aspects": [{"testAspect": {"key": "value"}}],
        }
    }

    emitter.emit(test_mce)
    assert len(emitter.get_emitted_mces()) == 1
    assert emitter.get_emitted_mces()[0] == test_mce


def test_get_datahub_emitter_dry_run(monkeypatch):
    monkeypatch.setenv("DATAHUB_DRY_RUN", "true")
    emitter = get_datahub_emitter()
    assert isinstance(emitter, DryRunEmitter)


def test_metadata_setup_register_types(tmp_path):
    # Create temporary test type file
    types_dir = tmp_path / "types"
    types_dir.mkdir()
    test_type = {
        "entityType": "testType",
        "aspectSpecs": [{"name": "testAspect", "version": 1}],
    }

    with open(types_dir / "test_type.json", "w") as f:
        json.dump(test_type, f)

    setup = MetadataSetup()
    setup.types_dir = types_dir
    setup.emitter = DryRunEmitter()
    setup.register_all_types()

    emitted = setup.emitter.get_emitted_mces()
    assert len(emitted) == 1
    assert emitted[0]["proposedSnapshot"]["urn"].endswith(f":testType")
    assert "aspectSpecs" in emitted[0]["proposedSnapshot"]["aspects"][0]
