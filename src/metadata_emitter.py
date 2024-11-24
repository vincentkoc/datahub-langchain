import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

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

class BaseMetadataEmitter:
    """Base class for metadata emission to DataHub"""

    def __init__(self, platform: str, gms_server: str = None):
        self.platform = platform
        self.emitter = get_datahub_emitter(gms_server)
        self.is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"

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

        # Create URN based on entity type
        if entity_type == "MLMODEL":
            urn = make_ml_model_urn(platform=self.platform, name=name, env="PROD")
            properties_class = MLModelPropertiesClass
            snapshot_class = MLModelSnapshotClass
        else:  # Default to DATASET for everything else (including flows and runs)
            urn = make_dataset_urn(platform=self.platform, name=name, env="PROD")
            properties_class = DatasetPropertiesClass
            snapshot_class = DatasetSnapshotClass

        # Format custom properties
        custom_properties = {
            "metadata": json.dumps(metadata),
            "sub_type": sub_type or "default",
            "platform_instance": self.platform,
            "created_at": datetime.now().isoformat()
        }

        if icon_url:
            custom_properties["logoUrl"] = icon_url

        if tags:
            custom_properties["tags"] = json.dumps(tags)

        # Create aspects
        status = StatusClass(removed=False)

        properties = properties_class(
            name=name,
            description=description,
            customProperties=custom_properties
        )

        browse_paths_aspect = BrowsePathsClass(paths=browse_paths)

        # Create MCE
        mce = MetadataChangeEventClass(
            proposedSnapshot=snapshot_class(
                urn=urn,
                aspects=[
                    status,
                    properties,
                    browse_paths_aspect
                ]
            )
        )

        # Add lineage if upstream URNs provided
        if upstream_urns:
            mce.proposedSnapshot.aspects.append({
                "com.linkedin.metadata.relationship.DownstreamOf": {
                    "downstreamOf": [
                        {"entity": {"urn": urn}} for urn in upstream_urns
                    ]
                }
            })

        # Debug output
        print(f"\nEmitting metadata for {name}")
        print(f"URN: {urn}")
        if self.is_dry_run:
            print("Metadata structure:")
            print(json.dumps(mce.to_obj(), indent=2))

        # Emit metadata
        self.emitter.emit(mce)
        print(f"Successfully emitted metadata for {name}")

        # Add delay if not in dry run
        if not self.is_dry_run:
            time.sleep(METADATA_PUSH_DELAY)

        return urn

    def emit_batch(self, items: List[Dict], batch_size: int = 5):
        """Emit metadata in batches"""
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            for item in batch:
                self.emit_metadata(**item)
            if not self.is_dry_run:
                time.sleep(METADATA_PUSH_DELAY * 2)
