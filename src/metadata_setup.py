import json
import os
from datetime import datetime
from pathlib import Path
from typing import Union

from dotenv import load_dotenv, find_dotenv
from datahub.emitter.mce_builder import make_dataset_urn
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent
from datahub.metadata.com.linkedin.pegasus2avro.metadata.snapshot import DatasetSnapshot
from datahub.metadata.schema_classes import (
    DatasetSnapshotClass,
    DatasetPropertiesClass,
    StatusClass,
    MetadataChangeEventClass,
)

# Find and load the .env file
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE, override=True)
    print(f"Loaded environment from: {ENV_FILE}")
else:
    print("Warning: No .env file found")

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
        print("\n=== DataHub Emitter Setup ===")
        print(f"GMS Server: {gms_server}")
        print(f"Token Present: {'yes' if token else 'no'}")

        # Add auth header to session
        if token:
            self._session.headers.update({
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            })
            print("Added auth headers")
            print(f"Current headers: {self._session.headers}")

    def emit(self, mce: Union[dict, MetadataChangeEventClass]) -> None:
        """Emit metadata with proper error handling"""
        try:
            print("\n=== Emitting Metadata ===")
            print(f"MCE Type: {type(mce)}")

            # Convert dict to MetadataChangeEvent if needed
            if isinstance(mce, dict):
                print(f"URN: {mce['proposedSnapshot']['urn']}")
                print("Converting dict to MetadataChangeEvent...")

                # Create DatasetSnapshot
                snapshot = DatasetSnapshotClass(
                    urn=mce["proposedSnapshot"]["urn"],
                    aspects=mce["proposedSnapshot"]["aspects"]
                )

                # Create MetadataChangeEvent
                mce = MetadataChangeEvent(
                    proposedSnapshot=snapshot
                )
                print("Conversion complete")

            # Emit the MCE
            super().emit(mce)
            print("Metadata emitted successfully")
        except Exception as e:
            print(f"Error emitting metadata: {str(e)}")
            raise Exception(f"Failed to emit metadata to DataHub: {str(e)}")


def get_datahub_emitter(
    gms_server: str = None,
) -> Union[DataHubEmitter, DryRunEmitter]:
    """Get the appropriate emitter based on configuration"""
    # Explicitly convert to boolean and handle case
    is_dry_run = str(os.getenv("DATAHUB_DRY_RUN", "false")).lower() == "true"

    # Get server and token from environment
    gms_server = gms_server or os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
    token = os.getenv("DATAHUB_TOKEN")

    print("\n=== DataHub Connection Details ===")
    print(f"GMS Server: {gms_server}")
    print(f"Token Present: {'yes' if token else 'no'}")
    print(f"Dry Run Mode: {is_dry_run}")

    try:
        if is_dry_run:
            print("Using DryRunEmitter")
            return DryRunEmitter()

        # Create emitter with token
        print("\nCreating DataHub emitter...")
        emitter = DataHubEmitter(gms_server=gms_server, token=token)

        # Test connection
        print("Testing connection...")
        response = emitter._session.get(f"{gms_server}/health")
        print(f"Health check response: {response.status_code}")
        print(f"Response content: {response.text}")

        if response.status_code != 200:
            raise Exception(f"Health check failed: {response.status_code}")

        return emitter
    except Exception as e:
        print(f"\nError connecting to DataHub: {str(e)}")
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

            # Create Status aspect
            status = StatusClass(removed=False)

            # Create Properties aspect
            properties = DatasetPropertiesClass(
                description=f"Custom type for {type_def['entityType']}",
                name=type_def["entityType"],
                customProperties={
                    "entityType": type_def["entityType"],
                    "schema": json.dumps(type_def["aspectSpecs"])
                }
            )

            # Create DatasetSnapshot
            snapshot = DatasetSnapshotClass(
                urn=type_urn,
                aspects=[status, properties]
            )

            # Create MetadataChangeEvent
            mce = MetadataChangeEventClass(
                proposedSnapshot=snapshot
            )

            print(f"\nRegistering type {type_def['entityType']}...")
            print(f"URN: {type_urn}")

            # Emit the metadata
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
