import json
import os
from pathlib import Path
from typing import Union
from datetime import datetime

from dotenv import load_dotenv, find_dotenv

from datahub.emitter.mce_builder import make_dataset_urn
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import (
    MetadataChangeEventClass,
    DatasetSnapshotClass,
    DatasetPropertiesClass,
)

# Find and load the .env file from the project root
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE, override=True)
    print(f"Loaded environment from: {ENV_FILE}")
else:
    print("Warning: No .env file found")

# Print environment variables for debugging
print(f"DATAHUB_DRY_RUN: {os.getenv('DATAHUB_DRY_RUN')}")
print(f"DATAHUB_GMS_URL: {os.getenv('DATAHUB_GMS_URL')}")
print(f"DATAHUB_TOKEN: {'Set' if os.getenv('DATAHUB_TOKEN') else 'Not set'}")


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

    def __init__(self, gms_server: str = None, token: str = None):
        super().__init__(gms_server=gms_server, token=token)
        # Add auth header to session
        if token:
            self._session.headers.update({"Authorization": f"Bearer {token}"})

    def emit(self, mce: Union[dict, MetadataChangeEventClass]) -> None:
        """Emit metadata with proper error handling"""
        try:
            # Test connection first
            self.test_connection()

            # Convert all URNs to dataset URNs
            if isinstance(mce, dict):
                urn = mce["proposedSnapshot"]["urn"]
                if not urn.startswith("urn:li:dataset:"):
                    # Convert custom URN to dataset URN
                    entity_type = urn.split(":")[2]
                    name = urn.split(":")[3]
                    mce["proposedSnapshot"]["urn"] = make_dataset_urn(
                        platform=entity_type,
                        name=name,
                        env="PROD"
                    )

            # Emit the MCE
            super().emit(mce)
        except Exception as e:
            raise Exception(f"Failed to emit metadata to DataHub: {str(e)}")


def get_datahub_emitter(
    gms_server: str = None,
) -> Union[DataHubEmitter, DryRunEmitter]:
    """Get the appropriate emitter based on configuration"""
    # Explicitly convert to boolean and handle case
    dry_run_value = os.getenv("DATAHUB_DRY_RUN", "false")
    is_dry_run = dry_run_value.lower() in ["true", "1", "yes", "on"]

    # Get server and token from environment
    gms_server = gms_server or os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
    token = os.getenv("DATAHUB_TOKEN")

    if not token and not is_dry_run:
        print("\nWarning: DATAHUB_TOKEN not set. Authentication may fail.")

    try:
        if is_dry_run:
            # Return DryRunEmitter without printing message
            return DryRunEmitter()

        # Create emitter with token
        emitter = DataHubEmitter(gms_server=gms_server, token=token)

        # Test connection
        emitter.test_connection()
        return emitter
    except Exception as e:
        if is_dry_run:
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

            # Create dataset URN for type registration
            type_urn = make_dataset_urn(
                platform="datahub",
                name=type_def['entityType'],
                env="PROD"
            )

            # Create a simpler metadata structure that matches DataHub's schema
            mce = MetadataChangeEventClass(
                proposedSnapshot=DatasetSnapshotClass(
                    urn=type_urn,
                    aspects=[
                        {
                            "com.linkedin.common.Status": {
                                "removed": False
                            }
                        },
                        {
                            "com.linkedin.common.DatasetProperties": {
                                "description": f"Custom type for {type_def['entityType']}",
                                "customProperties": {
                                    "entityType": type_def["entityType"],
                                    "schema": json.dumps(type_def["aspectSpecs"])
                                }
                            }
                        }
                    ]
                )
            )

            # Emit the MCE
            if isinstance(self.emitter, DryRunEmitter):
                # For dry run, convert to dict
                mce_dict = {
                    "proposedSnapshot": {
                        "urn": type_urn,
                        "aspects": [
                            {
                                "com.linkedin.common.Status": {
                                    "removed": False
                                }
                            },
                            {
                                "com.linkedin.common.DatasetProperties": {
                                    "description": f"Custom type for {type_def['entityType']}",
                                    "customProperties": {
                                        "entityType": type_def["entityType"],
                                        "schema": json.dumps(type_def["aspectSpecs"])
                                    }
                                }
                            }
                        ]
                    },
                    "systemMetadata": {
                        "lastObserved": int(datetime.now().timestamp() * 1000),
                        "runId": "metadata-setup"
                    }
                }
                self.emitter.emit(mce_dict)
            else:
                # For real emitter, use MetadataChangeEventClass
                self.emitter.emit(mce)

            print(f"Successfully registered type from {file_path.name}")
            return True
        except Exception as e:
            print(f"Error registering type from {file_path.name}: {str(e)}")
            if not isinstance(self.emitter, DryRunEmitter):
                raise
            return False


if __name__ == "__main__":
    setup = MetadataSetup()
    if not setup.register_all_types():
        print("\nWarning: Some types failed to register. Check the logs above for details.")
