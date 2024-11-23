import json
import os
from pathlib import Path
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent

class MetadataSetup:
    def __init__(self, gms_server="http://localhost:8080"):
        self.emitter = DatahubRestEmitter(gms_server)
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
