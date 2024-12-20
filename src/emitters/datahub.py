from typing import Dict, Any, Optional
from datetime import datetime
import json
import base64
import time
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.emitter.mce_builder import make_dataset_urn, make_ml_model_urn, make_tag_urn, make_data_job_urn
from datahub.metadata.schema_classes import (
    MLModelPropertiesClass,
    MLModelKeyClass,
    MLHyperParamClass,
    MetadataChangeEventClass,
    MLModelSnapshotClass,
    GlobalTagsClass,
    TagAssociationClass
)

from ..base import LLMMetadataEmitter, LLMModel, LLMRun, LLMChain
from ..config import ObservabilityConfig
from ..utils.pipeline_utils import detect_pipeline_name
from ..platforms.extender import DataHubPlatformExtender

class CustomDatahubRestEmitter(DatahubRestEmitter):
    """DataHub REST emitter with enhanced authentication and error handling"""

    def __init__(self, gms_server: str, token: Optional[str] = None, debug: bool = False):
        # Initialize without calling parent's __init__ to avoid header conflicts
        self._gms_server = gms_server
        self.debug = debug

        # Create new session
        self._session = requests.Session()

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
        self._session.timeout = 5

        # Clear any existing headers
        self._session.headers.clear()

        # Set minimal headers exactly matching the working curl command
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)"
        }

        if token:
            # Clean up token
            token = token.strip().strip("'").strip('"')

            if self.debug:
                self._debug("Token length:", len(token))
                self._debug("Token first 20 chars:", token[:20])
                self._debug("Token last 20 chars:", token[-20:])

            headers["Authorization"] = f"Bearer {token}"

        self._session.headers.update(headers)

        if self.debug:
            self._debug("Session headers after setup:", dict(self._session.headers))
            if token:
                auth_header = self._session.headers.get("Authorization", "")
                self._debug("Authorization header length:", len(auth_header))
                self._debug("Authorization header first 50 chars:", auth_header[:50])
                self._debug("Authorization header last 50 chars:", auth_header[-50:])

    def emit(self, *events: MetadataChangeEventClass) -> None:
        """Override emit to add debugging and proper request formatting"""
        if self.debug:
            self._debug("Emitting events with headers:", dict(self._session.headers))

        try:
            for event in events:
                url = f"{self._gms_server}/entities?action=ingest"

                # Add Content-Type header for the POST request
                headers = {"Content-Type": "application/json"}

                # Get the event object and extract the snapshot
                event_obj = event.to_obj()
                if "proposedSnapshot" not in event_obj:
                    raise ValueError("Event object must contain proposedSnapshot")

                snapshot = event_obj["proposedSnapshot"]

                # The snapshot type should be the first key in the snapshot dict
                snapshot_type = next(iter(snapshot.keys()))

                # Convert namespace to match the working version
                corrected_type = "com.linkedin.metadata.snapshot.MLModelSnapshot"

                # Format payload according to DataHub's expected schema
                payload = {
                    "entity": {
                        "value": {
                            corrected_type: {
                                "urn": snapshot[snapshot_type]["urn"],
                                "aspects": [
                                    {
                                        "com.linkedin.ml.metadata.MLModelProperties": {
                                            "customProperties": aspect.get("customProperties", {}),
                                            "description": aspect.get("description", ""),
                                            "type": aspect.get("type", "Language Model"),
                                            "hyperParams": [
                                                {
                                                    "name": param["name"],
                                                    "value": param["value"]
                                                }
                                                for param in aspect.get("hyperParams", [])
                                            ] if aspect.get("hyperParams") else []
                                        }
                                    }
                                    if "Properties" in aspect_type
                                    else {
                                        "com.linkedin.common.GlobalTags": {
                                            "tags": [
                                                {
                                                    "tag": tag["tag"]
                                                }
                                                for tag in aspect.get("tags", [])
                                            ]
                                        }
                                    }
                                    for aspect_obj in snapshot[snapshot_type]["aspects"]
                                    for aspect_type, aspect in aspect_obj.items()
                                ]
                            }
                        }
                    }
                }

                if self.debug:
                    self._debug("Request URL:", url)
                    self._debug("Request headers:", headers)
                    self._debug("Request payload:", payload)

                # Implement retry logic
                retries = 0
                max_retries = 3
                while retries < max_retries:
                    try:
                        response = self._session.post(url, json=payload, headers=headers)

                        if self.debug:
                            self._debug("Response status:", response.status_code)
                            self._debug("Response headers:", dict(response.headers))
                            if response.text:
                                self._debug("Response body:", response.text)

                        if response.status_code == 200:
                            break

                        retries += 1
                        if retries == max_retries:
                            raise Exception(f"Failed after {max_retries} retries: {response.text}")
                        time.sleep(0.2 * (2 ** retries))  # Exponential backoff

                    except Exception as e:
                        if retries == max_retries - 1:
                            if self.debug:
                                self._debug_error(e)
                            raise
                        retries += 1
                        time.sleep(0.2 * (2 ** retries))

        except Exception as e:
            if self.debug:
                self._debug_error(e)
            raise

    def _debug(self, msg: str, *args: Any) -> None:
        """Print debug message if debug mode is enabled"""
        if self.debug:
            print(f"\n=== DEBUG: {msg} ===")
            for arg in args:
                if isinstance(arg, dict):
                    # Ensure full output of long strings
                    print(json.dumps(arg, default=str, indent=2, ensure_ascii=False))
                else:
                    print(json.dumps(arg, default=str, ensure_ascii=False))

    def _debug_error(self, e: Exception) -> None:
        """Print detailed error information in debug mode"""
        self._debug("Error Details",
            f"Exception type: {type(e)}",
            f"Exception args: {e.args}"
        )
        if hasattr(e, 'response'):
            self._debug("Response Details",
                f"Status: {e.response.status_code}",
                f"Headers: {dict(e.response.headers)}",
                f"Body: {e.response.text}"
            )

class DataHubEmitter(LLMMetadataEmitter):
    """Emits LLM metadata to DataHub with comprehensive error handling and lineage tracking"""

    def __init__(self, gms_server: Optional[str] = None, debug: bool = False, hard_fail: bool = True):
        self.platform = "langchain"
        self.config = ObservabilityConfig.from_env()
        self.debug = debug
        self.hard_fail = hard_fail
        self._successful_emissions = set()

        # Clean up GMS server URL
        self.gms_server = gms_server or self.config.datahub_gms_url
        if self.gms_server:
            self.gms_server = self.gms_server.split("#")[0].strip()

        if self.debug:
            print(f"\nInitializing DataHub emitter with server: {self.gms_server}")

        # Initialize emitter with authentication
        self.emitter = CustomDatahubRestEmitter(
            gms_server=self.gms_server,
            token=self.config.datahub_token,
            debug=self.debug
        )

        # Initialize platform extender
        self.platform_extender = DataHubPlatformExtender(
            gms_server=self.gms_server,
            token=self.config.datahub_token
        )

    def _emit_with_retry(self, mce: MetadataChangeEventClass) -> None:
        """Emit metadata with proper error handling and retry"""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                if not self.config.datahub_dry_run:
                    self.emitter.emit(mce)
                    return  # Success, exit retry loop
            except Exception as e:
                if self.debug:
                    print(f"\n✗ Emission failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    if self.hard_fail:
                        raise
                else:
                    time.sleep(retry_delay * (attempt + 1))

    def emit_model(self, model: LLMModel) -> str:
        """Emit model metadata to DataHub"""
        try:
            model_urn = make_ml_model_urn(
                platform=self.platform,
                model_name=model.name,
                env="PROD"
            )

            # Create properties without group reference
            properties = MLModelPropertiesClass(
                description=f"{model.provider} {model.name} Language Model",
                type="Language Model",
                customProperties={
                    "provider": str(model.provider),
                    "model_family": str(model.model_family),
                    "capabilities": str(model.capabilities),
                    "parameters": str(model.parameters),
                    "raw_name": model.metadata.get("raw_name", model.name),
                    "source": "langchain"
                },
                hyperParams=[
                    MLHyperParamClass(name=k, value=str(v))
                    for k, v in model.parameters.items()
                ] if model.parameters else None
            )

            # Create the metadata change event
            mce = MetadataChangeEventClass(
                proposedSnapshot=MLModelSnapshotClass(
                    urn=model_urn,
                    aspects=[properties]
                )
            )

            self._emit_with_retry(mce)
            return model_urn

        except Exception as e:
            if self.debug:
                print(f"\n✗ Failed to emit model {model.name}: {e}")
            if self.hard_fail:
                raise
            return ""

    def emit_run(self, run: LLMRun) -> str:
        """Emit run metadata as part of a pipeline"""
        try:
            # Get pipeline name from metadata or detect from file
            pipeline_name = run.metadata.get("pipeline_name")
            if not pipeline_name:
                pipeline_name = detect_pipeline_name()

            # Create URN for the pipeline
            pipeline_urn = make_ml_model_urn(
                platform=self.platform,
                model_name=pipeline_name,
                env="PROD"
            )

            # Flatten metrics into simple key-value pairs
            metrics = {}
            for k, v in run.metrics.items():
                if isinstance(v, (int, float, bool)):
                    metrics[k] = str(v)
                elif isinstance(v, dict):
                    # For nested metrics like token_usage, flatten with underscore
                    for sub_k, sub_v in v.items():
                        metrics[f"{k}_{sub_k}"] = str(sub_v)
                else:
                    metrics[k] = str(v)

            # Create pipeline properties
            properties = MLModelPropertiesClass(
                description=f"LangChain Pipeline: {pipeline_name}",
                type="Pipeline",
                customProperties={
                    "pipeline_name": pipeline_name,
                    "framework": "langchain",
                    "last_run_id": str(run.id),
                    "last_run_time": str(run.start_time.isoformat()),
                    "last_run_status": "completed" if not run.metrics.get("error") else "failed",
                    **metrics
                }
            )

            # Add tags
            tags = GlobalTagsClass(
                tags=[
                    TagAssociationClass(tag=f"urn:li:tag:pipeline"),
                    TagAssociationClass(tag=f"urn:li:tag:langchain"),
                    TagAssociationClass(tag=f"urn:li:tag:{'success' if not run.metrics.get('error') else 'failed'}")
                ]
            )

            # Create the metadata change event
            mce = MetadataChangeEventClass(
                proposedSnapshot=MLModelSnapshotClass(
                    urn=pipeline_urn,
                    aspects=[properties, tags]
                )
            )

            # Add model lineage through custom properties
            if run.model:
                model_urn = self.emit_model(run.model)
                properties.customProperties["upstream_model"] = model_urn

            self._emit_with_retry(mce)
            return pipeline_urn

        except Exception as e:
            if self.debug:
                print(f"\n✗ Failed to emit pipeline run {run.id}: {e}")
            if self.hard_fail:
                raise
            return ""

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

    def emit(self, mce: MetadataChangeEventClass):
        """Emit a MetadataChangeEvent to DataHub"""
        try:
            self.emitter.emit(mce)
        except Exception as e:
            if self.debug:
                print(f"Error emitting MCE: {e}")
            if self.hard_fail:
                raise

    def register_platforms(self) -> None:
        """Register all supported platforms in DataHub"""
        try:
            self.platform_extender.register_all_platforms()
        except Exception as e:
            if self.debug:
                print(f"Failed to register platforms: {e}")
            if self.hard_fail:
                raise
