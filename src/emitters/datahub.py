from typing import Dict, Any, Optional
from datetime import datetime
import json

from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.emitter.mce_builder import make_dataset_urn, make_ml_model_urn
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    MLModelPropertiesClass,
    StatusClass,
    MetadataChangeEventClass,
    DatasetSnapshotClass,
    MLModelSnapshotClass,
)

from .base import BaseEmitter
from ..base import LLMMetadataEmitter, LLMModel, LLMRun, LLMChain

class DataHubEmitter(BaseEmitter, LLMMetadataEmitter):
    """Emits LLM metadata to DataHub"""

    def __init__(self, gms_server: Optional[str] = None):
        super().__init__("llm", gms_server)

    def emit_model(self, model: LLMModel) -> str:
        """Emit model metadata to DataHub"""
        try:
            model_urn = make_ml_model_urn(
                platform=self.platform,
                name=f"{model.provider}/{model.name}",
                env="PROD"
            )

            if model_urn in self._emitted_urns:
                print(f"\n⚠ Model already emitted: {model.name}")
                return model_urn

            # Simplify and flatten the properties
            properties = {
                "provider": str(model.provider),
                "model_family": str(model.model_family),
                "capabilities": str(model.capabilities),
                "parameters": str(model.parameters),
                "metadata": str(model.metadata)
            }

            mce = MetadataChangeEventClass(
                proposedSnapshot=MLModelSnapshotClass(
                    urn=model_urn,
                    aspects=[
                        MLModelPropertiesClass(
                            description=f"{model.provider} {model.name} Language Model",
                            customProperties=properties
                        )
                    ]
                )
            )

            self._emit_with_retry(mce)
            self._emitted_urns.add(model_urn)
            return model_urn

        except Exception as e:
            print(f"\n✗ Failed to emit model {model.name}: {e}")
            raise

    def emit_run(self, run: LLMRun) -> str:
        """Emit run metadata to DataHub"""
        try:
            # Simplify and flatten the properties
            properties = {
                "run_id": str(run.id),
                "inputs": str(run.inputs),
                "outputs": str(run.outputs),
                "metrics": str(run.metrics),
                "start_time": str(run.start_time.isoformat()),
                "end_time": str(run.end_time.isoformat() if run.end_time else None),
                "parent_id": str(run.parent_id or "none"),
                "model": str(run.model.name if run.model else "unknown"),
                "status": "completed" if not run.metrics.get("error") else "failed"
            }

            run_urn = make_dataset_urn(
                platform=self.platform,
                name=f"runs/{run.id}",
                env="PROD"
            )

            mce = MetadataChangeEventClass(
                proposedSnapshot=DatasetSnapshotClass(
                    urn=run_urn,
                    aspects=[
                        DatasetPropertiesClass(
                            description=f"LLM Run {run.id}",
                            customProperties=properties
                        )
                    ]
                )
            )

            self._emit_with_retry(mce)

            # Emit lineage to model if available
            if run.model:
                try:
                    model_urn = self.emit_model(run.model)
                    self.emit_lineage(run_urn, model_urn, "Uses")
                except Exception as e:
                    print(f"✗ Failed to emit lineage for run {run.id}: {e}")

            return run_urn

        except Exception as e:
            print(f"\n✗ Failed to emit run {run.id}: {e}")
            raise

    def emit_chain(self, chain: LLMChain) -> str:
        """Emit chain metadata to DataHub"""
        try:
            # Simplify and flatten the properties
            properties = {
                "chain_id": str(chain.id),
                "name": str(chain.name),
                "components": str(chain.components),
                "config": str(chain.config),
                "metadata": str(chain.metadata)
            }

            chain_urn = make_dataset_urn(
                platform=self.platform,
                name=f"chains/{chain.id}",
                env="PROD"
            )

            mce = MetadataChangeEventClass(
                proposedSnapshot=DatasetSnapshotClass(
                    urn=chain_urn,
                    aspects=[
                        DatasetPropertiesClass(
                            description=f"LLM Chain {chain.name}",
                            customProperties=properties
                        )
                    ]
                )
            )

            self._emit_with_retry(mce)
            return chain_urn

        except Exception as e:
            print(f"\n✗ Failed to emit chain {chain.name}: {e}")
            raise
