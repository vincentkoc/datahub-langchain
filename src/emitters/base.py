import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from datahub.emitter.mce_builder import make_dataset_urn, make_ml_model_urn
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    MLModelPropertiesClass,
    StatusClass,
    MetadataChangeEventClass,
    DatasetSnapshotClass,
    MLModelSnapshotClass,
)

class BaseEmitter:
    """Base class for metadata emission"""

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    TIMEOUT = 5  # seconds

    def __init__(self, platform: str, gms_server: Optional[str] = None):
        from ..config import ObservabilityConfig
        self.platform = platform
        self.config = ObservabilityConfig.from_env()
        self._setup_emitter(gms_server)
        self._emitted_urns = set()

    def _setup_emitter(self, gms_server: Optional[str]) -> None:
        """Setup the DataHub emitter with retry configuration"""
        from datahub.emitter.rest_emitter import DatahubRestEmitter

        self.gms_server = gms_server or self.config.datahub_gms_url
        self.emitter = DatahubRestEmitter(self.gms_server)

        # Configure session with optimized retries and timeouts
        if hasattr(self.emitter, '_session'):
            retry_strategy = Retry(
                total=self.MAX_RETRIES,
                backoff_factor=0.2,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.emitter._session.mount("http://", adapter)
            self.emitter._session.mount("https://", adapter)
            self.emitter._session.timeout = self.TIMEOUT

    def _emit_with_retry(self, mce: MetadataChangeEventClass) -> None:
        """Emit metadata with retry logic"""
        if self.config.datahub_dry_run:
            print("\n=== Dry Run - Would emit MCE ===")
            print(mce.to_obj())
            return

        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            try:
                if hasattr(self.emitter, '_session'):
                    self.emitter._session.timeout = self.TIMEOUT

                self.emitter.emit(mce)
                return
            except Exception as e:
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))

        raise Exception(f"Failed to emit metadata after {self.MAX_RETRIES} attempts: {last_exception}")

    def emit_metadata(
        self,
        name: str,
        metadata: Dict[str, Any],
        description: str,
        browse_paths: List[str],
        entity_type: str = "DATASET",
        sub_type: Optional[str] = None,
        upstream_urns: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Emit metadata with consistent structure"""
        # Generate URN
        if entity_type == "MLMODEL":
            urn = make_ml_model_urn(self.platform, name, "PROD")
        else:
            urn = make_dataset_urn(self.platform, name, "PROD")

        # Add source tracking
        metadata["source"] = {
            "platform": self.platform,
            "sub_type": sub_type,
            "ingestion_time": datetime.now().isoformat()
        }

        # Create aspects
        aspects = [
            StatusClass(removed=False),
            DatasetPropertiesClass(
                description=description,
                customProperties={
                    **metadata,
                    "sub_type": sub_type or "default",
                    "tags": tags or []
                }
            ) if entity_type == "DATASET" else MLModelPropertiesClass(
                description=description,
                customProperties=metadata
            )
        ]

        # Create MCE
        mce = MetadataChangeEventClass(
            proposedSnapshot=(
                DatasetSnapshotClass if entity_type == "DATASET" else MLModelSnapshotClass
            )(
                urn=urn,
                aspects=aspects
            )
        )

        # Emit with retry
        self._emit_with_retry(mce)

        # Track emitted URN
        self._emitted_urns.add(urn)

        return urn

    def emit_lineage(self, source_urn: str, target_urn: str, lineage_type: str) -> None:
        """Emit lineage relationship"""
        if self.config.datahub_dry_run:
            print(f"\n=== Dry Run - Would emit lineage ===")
            print(f"Source: {source_urn}")
            print(f"Target: {target_urn}")
            print(f"Type: {lineage_type}")
            return

        try:
            # Create upstream lineage
            upstream_mce = self._create_lineage_mce(source_urn, target_urn, lineage_type)
            self._emit_with_retry(upstream_mce)

            # Create downstream lineage
            downstream_mce = self._create_lineage_mce(target_urn, source_urn, lineage_type)
            self._emit_with_retry(downstream_mce)

        except Exception as e:
            print(f"Failed to emit lineage: {e}")
            if not self.config.datahub_dry_run:
                raise

    def _create_lineage_mce(self, source_urn: str, target_urn: str, lineage_type: str) -> MetadataChangeEventClass:
        """Create lineage MCE"""
        source_is_dataset = "dataset" in source_urn
        target_is_dataset = "dataset" in target_urn

        return MetadataChangeEventClass(
            proposedSnapshot=(
                DatasetSnapshotClass if source_is_dataset else MLModelSnapshotClass
            )(
                urn=source_urn,
                aspects=[{
                    "com.linkedin.metadata.aspect.UpstreamLineage": {
                        "upstreams": [{
                            "auditStamp": {
                                "time": int(datetime.now().timestamp() * 1000),
                                "actor": "urn:li:corpuser:datahub"
                            },
                            "dataset": {
                                "entityType": "dataset" if target_is_dataset else "mlModel",
                                "urn": target_urn
                            },
                            "type": lineage_type
                        }]
                    }
                }]
            )
        )
