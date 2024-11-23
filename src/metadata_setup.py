import json
import os
from pathlib import Path
from typing import Union

from datahub.emitter.mce_builder import make_dataset_urn
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import (
    MetadataChangeEventClass,
    DatasetSnapshotClass,
    DatasetPropertiesClass,
)


class DryRunEmitter:
    """Emitter that prints metadata changes instead of sending them to DataHub"""

    def __init__(self):
        self._emitted_mces = []

    def emit(self, mce: Union[dict, MetadataChangeEventClass]) -> None:
        """Print the metadata change event instead of emitting it"""
        try:
            if isinstance(mce, dict):
                mce_dict = mce
            else:
                mce_dict = mce.to_obj()

            self._emitted_mces.append(mce_dict)
            print("\n=== Metadata Change Event ===")

            # Handle both dictionary formats
            if "proposedSnapshot" in mce_dict:
                urn = mce_dict["proposedSnapshot"].get("urn")
                aspects = mce_dict["proposedSnapshot"].get("aspects", [])
                print(f"URN: {urn}")

                for aspect in aspects:
                    if isinstance(aspect, dict):
                        for aspect_name, aspect_value in aspect.items():
                            print(f"\nAspect: {aspect_name}")
                            print(json.dumps(aspect_value, indent=2))

            print("\n===========================\n")
        except Exception as e:
            print(f"Error in dry run emission: {str(e)}")
            print("Metadata content:")
            print(json.dumps(mce_dict, indent=2))

    def get_emitted_mces(self) -> list:
        """Return all emitted MCEs"""
        return self._emitted_mces.copy()


class DataHubEmitter(DatahubRestEmitter):
    """Extended DataHub emitter with better error handling"""

    def emit(self, mce: Union[dict, MetadataChangeEventClass]) -> None:
        """Emit metadata with proper error handling"""
        try:
            # Test connection first
            self.test_connection()

            # Convert dict to MCE if needed
            if isinstance(mce, dict):
                snapshot = DatasetSnapshotClass(
                    urn=mce["proposedSnapshot"]["urn"],
                    aspects=mce["proposedSnapshot"]["aspects"]
                )
                mce = MetadataChangeEventClass(
                    proposedSnapshot=snapshot
                )

            # Emit the MCE
            super().emit(mce)
        except Exception as e:
            raise Exception(f"Failed to emit metadata to DataHub: {str(e)}")


def get_datahub_emitter(
    gms_server: str = None,
) -> Union[DataHubEmitter, DryRunEmitter]:
    """Get the appropriate emitter based on configuration"""
    is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"

    if is_dry_run:
        print(
            "\nRunning in DRY RUN mode - metadata will be printed but not sent to DataHub"
        )
        return DryRunEmitter()

    gms_server = gms_server or os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
    token = os.getenv("DATAHUB_TOKEN")

    try:
        emitter = DataHubEmitter(gms_server=gms_server, token=token)
        emitter.test_connection()
        return emitter
    except Exception as e:
        if is_dry_run:
            print(f"\nWarning: Could not connect to DataHub ({str(e)})")
            print("Continuing in dry run mode - metadata will be printed but not sent to DataHub")
            return DryRunEmitter()
        else:
            raise Exception(
                f"Could not connect to DataHub at {gms_server}. "
                f"Error: {str(e)}. "
                "Set DATAHUB_DRY_RUN=true to run without DataHub connection."
            )


class MetadataSetup:
    def __init__(self, gms_server: str = None):
        self.emitter = get_datahub_emitter(gms_server)
        self.types_dir = Path(__file__).parent.parent / "metadata" / "types"

    def register_all_types(self):
        """Register all custom types defined in metadata/types directory"""
        success = True
        for type_file in self.types_dir.glob("*.json"):
            if not self.register_type_from_file(type_file):
                success = False
        return success

    def register_type_from_file(self, file_path) -> bool:
        """Register a single type from a JSON file"""
        try:
            with open(file_path) as f:
                type_def = json.load(f)

            # Create URN for the type using dataset URN format
            type_urn = make_dataset_urn(
                platform="datahub",
                name=f"entityType_{type_def['entityType']}",
                env="PROD"
            )

            # Create properties aspect
            properties = {
                "DatasetProperties": {
                    "name": type_def["entityType"],
                    "description": f"Custom type for {type_def['entityType']}",
                    "customProperties": {
                        "aspectSpecs": json.dumps(type_def["aspectSpecs"])
                    }
                }
            }

            # Create MCE dict directly
            mce_dict = {
                "proposedSnapshot": {
                    "urn": type_urn,
                    "aspects": [properties]
                }
            }

            self.emitter.emit(mce_dict)
            print(f"Successfully registered type from {file_path.name}")
            return True
        except Exception as e:
            print(f"Error registering type from {file_path.name}: {str(e)}")
            return False


if __name__ == "__main__":
    setup = MetadataSetup()
    if not setup.register_all_types():
        print("\nWarning: Some types failed to register. Check the logs above for details.")
