from typing import Dict, Any, Optional
from datetime import datetime
import json
import base64
import time
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

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
from ..config import ObservabilityConfig

class CustomDatahubRestEmitter(DatahubRestEmitter):
    """Custom DatahubRestEmitter that properly handles authentication"""

    def __init__(self, gms_server: str, token: Optional[str] = None, debug: bool = False):
        super().__init__(gms_server=gms_server)
        self.debug = debug

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.2,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.timeout = 5  # 5 second timeout

        # Set up authentication
        if token:
            self._session.headers.update({
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-RestLi-Protocol-Version": "2.0.0"
            })
            if self.debug:
                self._debug("Session headers after auth setup:", self._session.headers)

    def emit(self, *events: MetadataChangeEventClass) -> None:
        if self.debug:
            self._debug("Emitting events with headers:", self._session.headers)
        try:
            super().emit(*events)
        except Exception as e:
            if self.debug:
                self._debug_error(e)
            raise

    def _debug(self, msg: str, *args: Any) -> None:
        """Print debug message if debug mode is enabled"""
        if self.debug:
            print(f"\n=== DEBUG: {msg} ===")
            for arg in args:
                print(json.dumps(arg, default=str, indent=2))

    def _debug_error(self, e: Exception) -> None:
        """Print detailed error information in debug mode"""
        self._debug("Error Details",
            f"Exception type: {type(e)}",
            f"Exception args: {e.args}"
        )
        if hasattr(e, 'response'):
            self._debug("Response Details",
                f"Status: {e.response.status_code}",
                f"Headers: {e.response.headers}",
                f"Body: {e.response.text}"
            )

class DataHubEmitter(BaseEmitter, LLMMetadataEmitter):
    """Emits LLM metadata to DataHub"""

    def __init__(self, gms_server: Optional[str] = None, debug: bool = False, hard_fail: bool = True):
        """Initialize DataHub emitter

        Args:
            gms_server: Optional GMS server URL
            debug: Enable debug logging
            hard_fail: If True, fail on first error. If False, continue processing
        """
        self.platform = "llm"
        self.config = ObservabilityConfig.from_env()
        self.debug = debug
        self.hard_fail = hard_fail

        # Clean up GMS server URL
        self.gms_server = gms_server or self.config.datahub_gms_url
        if self.gms_server:
            self.gms_server = self.gms_server.split("#")[0].strip()

        if self.debug:
            print(f"\nInitializing DataHub emitter with server: {self.gms_server}")
            print(f"Token from config: {'present' if self.config.datahub_token else 'missing'}")

        # Initialize emitter with authentication
        self.emitter = CustomDatahubRestEmitter(
            gms_server=self.gms_server,
            token=self.config.datahub_token,
            debug=self.debug
        )

        self._emitted_urns = set()

    def _emit_with_retry(self, mce: MetadataChangeEventClass) -> None:
        """Emit metadata with proper error handling"""
        try:
            if not self.config.datahub_dry_run:
                self.emitter.emit(mce)
        except Exception as e:
            if self.debug:
                print(f"\n✗ Emission failed: {str(e)}")
            if self.hard_fail or not self.config.datahub_dry_run:
                raise

    def emit_model(self, model: LLMModel) -> str:
        """Emit model metadata to DataHub"""
        try:
            model_urn = make_ml_model_urn(
                platform=self.platform,
                name=f"{model.provider}/{model.name}",
                env="PROD"
            )

            if model_urn in self._emitted_urns:
                if self.debug:
                    print(f"\n⚠ Model already emitted: {model.name}")
                return model_urn

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
            if self.debug:
                print(f"\n✗ Failed to emit model {model.name}: {e}")
            raise

    def emit_run(self, run: LLMRun) -> str:
        """Emit run metadata to DataHub"""
        try:
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
                    if self.debug:
                        print(f"✗ Failed to emit lineage for run {run.id}: {e}")
                    if self.hard_fail:
                        raise

            return run_urn

        except Exception as e:
            if self.debug:
                print(f"\n✗ Failed to emit run {run.id}: {e}")
            raise

    def emit_chain(self, chain: LLMChain) -> str:
        """Emit chain metadata to DataHub"""
        try:
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
            if self.debug:
                print(f"\n✗ Failed to emit chain {chain.name}: {e}")
            raise

    def emit_lineage(self, source_urn: str, target_urn: str, lineage_type: str) -> None:
        """Emit lineage between entities"""
        if self.config.datahub_dry_run:
            if self.debug:
                print(f"\n=== Dry Run - Would emit lineage ===")
                print(f"Source: {source_urn}")
                print(f"Target: {target_urn}")
                print(f"Type: {lineage_type}")
            return
