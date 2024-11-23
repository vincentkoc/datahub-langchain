import json
import os
from pathlib import Path
from typing import Union
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent

class DryRunEmitter:
    """Emitter that prints metadata changes instead of sending them to DataHub"""

    def __init__(self):
        self.emitted_mces = []

    def emit(self, mce: dict) -> None:
        """Print the metadata change event instead of emitting it"""
        self.emitted_mces.append(mce)
        print("\n=== Metadata Change Event ===")
        print(f"URN: {mce['proposedSnapshot']['urn']}")

        for aspect in mce['proposedSnapshot']['aspects']:
            aspect_name = list(aspect.keys())[0]
            print(f"\nAspect: {aspect_name}")
            print(json.dumps(aspect[aspect_name], indent=2))

        print("\n===========================\n")

    def get_emitted_mces(self):
        """Return all emitted MCEs"""
        return self.emitted_mces

def get_datahub_emitter(gms_server: str = None) -> Union[DatahubRestEmitter, DryRunEmitter]:
    """Get the appropriate emitter based on configuration"""
    is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"

    if is_dry_run:
        print("Running in DRY RUN mode - metadata will be printed but not sent to DataHub")
        return DryRunEmitter()

    gms_server = gms_server or os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
    token = os.getenv("DATAHUB_TOKEN")

    return DatahubRestEmitter(gms_server=gms_server, token=token)

class MetadataSetup:
    def __init__(self, gms_server: str = None):
        self.emitter = get_datahub_emitter(gms_server)
        self.types_dir = Path(__file__).parent.parent / "metadata" / "types"

    def register_all_types(self):
        """Register all custom types defined in metadata/types directory"""
        for type_file in self.types_dir.glob("*.json"):
            self.register_type_from_file(type_file)

    def register_type_from_file(self, file_path):
        """Register a single type from a JSON file"""
        with open(file_path) as f:
            type_def = json.load(f)

        mce = MetadataChangeEvent(
            proposedSnapshot={
                "entityType": type_def["entityType"],
                "aspectSpecs": type_def["aspectSpecs"]
            }
        )

        try:
            self.emitter.emit(mce)
            print(f"Successfully registered type from {file_path.name}")
        except Exception as e:
            print(f"Error registering type from {file_path.name}: {str(e)}")

if __name__ == "__main__":
    setup = MetadataSetup()
    setup.register_all_types()
