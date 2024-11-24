import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from datahub.emitter.mce_builder import (
    make_dataset_urn,
    make_ml_model_urn
)
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    MLModelPropertiesClass,
    StatusClass,
    BrowsePathsClass,
    DatasetSnapshotClass,
    MLModelSnapshotClass,
    MetadataChangeEventClass,
)

from src.metadata_setup import get_datahub_emitter

METADATA_PUSH_DELAY = 2.0
EMIT_TIMEOUT = 10  # seconds

class BaseMetadataEmitter:
    """Base class for metadata emission to DataHub"""

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    TIMEOUT = 10  # seconds

    def __init__(self, platform: str, gms_server: str = None):
        self.platform = platform
        self.emitter = get_datahub_emitter(gms_server)
        self.is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"

        # Configure session with retries and timeouts
        if hasattr(self.emitter, '_session'):
            retry_strategy = Retry(
                total=self.MAX_RETRIES,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.emitter._session.mount("http://", adapter)
            self.emitter._session.mount("https://", adapter)
            self.emitter._session.timeout = self.TIMEOUT

    def _emit_with_retry(self, mce: MetadataChangeEventClass, max_retries: int = None) -> None:
        """Emit metadata with retry logic and timeout"""
        retries = max_retries or self.MAX_RETRIES
        last_exception = None
        backoff = 1

        try:
            # Set timeout for emission
            if hasattr(self.emitter, '_session'):
                self.emitter._session.timeout = self.TIMEOUT

            # Create entity first with minimal aspects
            initial_mce = MetadataChangeEventClass(
                proposedSnapshot=mce.proposedSnapshot.__class__(
                    urn=mce.proposedSnapshot.urn,
                    aspects=[
                        aspect for aspect in mce.proposedSnapshot.aspects
                        if isinstance(aspect, (StatusClass, BrowsePathsClass))
                    ]
                )
            )

            # Emit initial entity
            print("\nCreating entity...")
            self.emitter.emit(initial_mce)
            time.sleep(1)  # Give time for entity to be created

            # Then update with properties
            properties_mce = MetadataChangeEventClass(
                proposedSnapshot=mce.proposedSnapshot.__class__(
                    urn=mce.proposedSnapshot.urn,
                    aspects=[
                        aspect for aspect in mce.proposedSnapshot.aspects
                        if not isinstance(aspect, (StatusClass, BrowsePathsClass))
                    ]
                )
            )

            # Emit properties
            print("Updating properties...")
            self.emitter.emit(properties_mce)
            time.sleep(1)  # Give time for properties to be updated

            # Finally, add lineage if present
            if any(isinstance(aspect, dict) and "com.linkedin.metadata.aspect.DownstreamLineage" in aspect
                   for aspect in mce.proposedSnapshot.aspects):
                lineage_mce = MetadataChangeEventClass(
                    proposedSnapshot=mce.proposedSnapshot.__class__(
                        urn=mce.proposedSnapshot.urn,
                        aspects=[
                            aspect for aspect in mce.proposedSnapshot.aspects
                            if isinstance(aspect, dict) and "com.linkedin.metadata.aspect.DownstreamLineage" in aspect
                        ]
                    )
                )
                print("Adding lineage...")
                self.emitter.emit(lineage_mce)

            return  # Success

        except Exception as e:
            last_exception = e
            print(f"Emission failed: {str(e)}")
            if not self.is_dry_run:
                raise Exception(f"Failed to emit metadata: {str(last_exception)}")

    def emit_metadata(
        self,
        name: str,
        metadata: Dict,
        description: str,
        browse_paths: List[str],
        entity_type: str = "DATASET",
        sub_type: str = None,
        upstream_urns: List[str] = None,
        tags: List[str] = None,
        icon_url: str = None
    ) -> str:
        """Emit metadata with consistent structure"""
        try:
            # Create URN based on entity type
            if entity_type == "MLMODEL":
                urn = make_ml_model_urn(
                    platform=self.platform,
                    model_name=name,
                    env="PROD"
                )
                properties_class = MLModelPropertiesClass
                snapshot_class = MLModelSnapshotClass
            else:  # Default to DATASET for everything else
                urn = make_dataset_urn(platform=self.platform, name=name, env="PROD")
                properties_class = DatasetPropertiesClass
                snapshot_class = DatasetSnapshotClass

            # Format custom properties with proper serialization
            custom_properties = {}
            for key, value in {
                "metadata": metadata,
                "sub_type": sub_type or "default",
                "platform_instance": self.platform,
                "created_at": datetime.now().isoformat(),
                "logoUrl": icon_url,
                "tags": tags,
                "name": name
            }.items():
                if value is not None:
                    if isinstance(value, (dict, list)):
                        custom_properties[key] = json.dumps(value, default=str)
                    else:
                        custom_properties[key] = str(value)

            # Create aspects
            aspects = []

            # Add Status aspect
            aspects.append(StatusClass(removed=False))

            # Add Properties aspect - remove name from direct constructor args
            aspects.append(properties_class(
                description=description,
                customProperties=custom_properties
            ))

            # Add BrowsePaths aspect
            aspects.append(BrowsePathsClass(paths=browse_paths))

            # Create MCE
            mce = MetadataChangeEventClass(
                proposedSnapshot=snapshot_class(
                    urn=urn,
                    aspects=aspects
                )
            )

            # Add lineage if upstream URNs provided
            if upstream_urns:
                aspects.append({
                    "com.linkedin.metadata.aspect.DownstreamLineage": {
                        "downstreams": [
                            {
                                "dataset": {"urn": upstream_urn},
                                "type": "TRANSFORMED",
                                "auditStamp": {
                                    "time": int(datetime.now().timestamp() * 1000),
                                    "actor": "urn:li:corpuser:datahub"
                                }
                            } for upstream_urn in upstream_urns
                        ]
                    }
                })

            # Debug output
            print(f"\nEmitting metadata for {name}")
            print(f"URN: {urn}")
            if self.is_dry_run:
                print("Metadata structure:")
                print(json.dumps(mce.to_obj(), indent=2))

            # Emit metadata with retry
            if not self.is_dry_run:
                try:
                    self._emit_with_retry(mce)
                    print(f"Successfully emitted metadata for {name}")
                except Exception as e:
                    print(f"Error emitting metadata: {str(e)}")
                    raise

            # Add delay between emissions
            if not self.is_dry_run:
                time.sleep(METADATA_PUSH_DELAY)

            return urn

        except Exception as e:
            print(f"Failed to emit metadata for {name}: {str(e)}")
            raise

    def emit_batch(self, items: List[Dict], batch_size: int = 5):
        """Emit metadata in batches with timeout handling"""
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            for item in batch:
                try:
                    self.emit_metadata(**item)
                except Exception as e:
                    print(f"Failed to emit batch item: {str(e)}")
                    if not self.is_dry_run:
                        raise
            if not self.is_dry_run:
                time.sleep(METADATA_PUSH_DELAY * 2)
