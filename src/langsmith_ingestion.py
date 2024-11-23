import json
import os
from datetime import datetime, timedelta
from uuid import UUID

from dotenv import load_dotenv
from langsmith import Client
from datahub.emitter.mce_builder import make_dataset_urn
from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent
from datahub.metadata.schema_classes import DatasetSnapshotClass

from src.metadata_setup import get_datahub_emitter, DryRunEmitter

load_dotenv()


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle UUID objects"""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


class LangSmithIngestion:
    def __init__(self, gms_server: str = None):
        # Default to GMS endpoint if not specified
        if not gms_server:
            gms_server = os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
            # If using frontend, append /api/gms
            if ":9002" in gms_server:
                gms_server = f"{gms_server}/api/gms"
            print(f"\nDEBUG: Using GMS endpoint: {gms_server}")

        self.emitter = get_datahub_emitter(gms_server)
        self.client = Client()
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "default")

    @property
    def is_dry_run(self) -> bool:
        """Determine dry run mode from emitter type"""
        return isinstance(self.emitter, DryRunEmitter)

    def emit_metadata(self, mce_dict: dict) -> str:
        """Emit metadata with dry run handling"""
        try:
            if self.is_dry_run:
                print(f"\nDEBUG: Would emit metadata:")
                print(json.dumps(mce_dict, indent=2))
                self.emitter.emit(mce_dict)
                return mce_dict["proposedSnapshot"]["urn"]

            # For live mode, convert to MetadataChangeEvent
            mce = MetadataChangeEvent(
                proposedSnapshot=DatasetSnapshotClass(
                    urn=mce_dict["proposedSnapshot"]["urn"],
                    aspects=[
                        {
                            "com.linkedin.common.Status": {
                                "removed": False
                            }
                        },
                        {
                            "com.linkedin.common.DatasetProperties": {
                                "description": mce_dict["proposedSnapshot"]["aspects"][1]["com.linkedin.common.DatasetProperties"]["description"],
                                "customProperties": mce_dict["proposedSnapshot"]["aspects"][1]["com.linkedin.common.DatasetProperties"]["customProperties"],
                                "name": mce_dict["proposedSnapshot"]["aspects"][1]["com.linkedin.common.DatasetProperties"]["name"]
                            }
                        }
                    ]
                )
            )
            self.emitter.emit(mce)
            return mce_dict["proposedSnapshot"]["urn"]
        except Exception as e:
            error_msg = f"Failed to emit metadata: {str(e)}"
            print(f"\nDEBUG: Error details: {error_msg}")
            print(f"DEBUG: Current metadata structure:")
            print(json.dumps(mce_dict, indent=2))
            if self.is_dry_run:
                print(f"DRY RUN ERROR: {error_msg}")
            else:
                raise Exception(error_msg)
            return None

    def emit_run_metadata(self, run):
        """Emit metadata for a single LangSmith run"""
        try:
            print(f"\nDEBUG: Processing run {run.id}")

            # Clean up error message if present
            error_message = None
            if hasattr(run, "error") and run.error:
                error_str = str(run.error)
                error_message = error_str.split("\n")[0] if "\n" in error_str else error_str

            # Extract token usage if available
            token_usage = {}
            if hasattr(run, "execution_metadata") and run.execution_metadata:
                token_usage = run.execution_metadata.get("token_usage", {})

            # Create run data with proper serialization
            run_data = {
                "runId": str(run.id),
                "name": str(getattr(run, "name", "")),
                "startTime": str(run.start_time),
                "endTime": str(run.end_time),
                "status": str(run.status),
                "inputs": dict(run.inputs),
                "outputs": dict(run.outputs) if run.outputs else None,
                "error": error_message,
                "runtime": float(getattr(run, "runtime_seconds", 0)),
                "parentRunId": str(run.parent_run_id) if getattr(run, "parent_run_id", None) else None,
                "childRunIds": [str(id) for id in getattr(run, "child_run_ids", [])],
                "tags": list(getattr(run, "tags", [])),
                "feedback": [],  # Empty list for now
                "metrics": {
                    "tokenUsage": {
                        "promptTokens": token_usage.get("prompt_tokens"),
                        "completionTokens": token_usage.get("completion_tokens"),
                        "totalTokens": token_usage.get("total_tokens"),
                    },
                    "latency": float(getattr(run, "latency", 0)),
                    "cost": float(getattr(run, "cost", 0)),
                },
            }

            # Use dataset URN format
            run_urn = make_dataset_urn(
                platform="llm",
                name=f"run_{run.id}",
                env="PROD"
            )

            # Create metadata with proper structure
            metadata = {
                "proposedSnapshot": {
                    "urn": run_urn,
                    "aspects": [
                        {
                            "com.linkedin.common.Status": {
                                "removed": False
                            }
                        },
                        {
                            "com.linkedin.common.DatasetProperties": {
                                "description": f"LangSmith Run {run.id}",
                                "customProperties": run_data,
                                "name": str(run.id)
                            }
                        }
                    ]
                },
                "systemMetadata": {
                    "lastObserved": int(datetime.now().timestamp() * 1000),
                    "runId": "langsmith-ingestion"
                }
            }

            print(f"\nDEBUG: Emitting metadata:")
            print(json.dumps(metadata, indent=2))

            return self.emit_metadata(metadata)
        except Exception as e:
            print(f"\nDEBUG: Error in emit_run_metadata: {str(e)}")
            print(f"Error processing run {run.id}: {str(e)}")
            return None

    def ingest_recent_runs(self, limit=100, days_ago=7):
        """Ingest metadata from recent LangSmith runs"""
        # Get runs from the last X days
        start_time = datetime.now() - timedelta(days=days_ago)

        try:
            # List runs with proper filters
            runs = self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                execution_order=1,  # Get top-level runs
                limit=limit,
            )

            run_urns = []
            for run in runs:
                try:
                    run_urn = self.emit_run_metadata(run)
                    if run_urn:
                        run_urns.append(run_urn)
                        if self.is_dry_run:
                            print(f"DRY RUN: Successfully processed run: {run.id}")
                        else:
                            print(f"Ingested run: {run.id}")
                except Exception as e:
                    print(f"Error processing run {run.id}: {str(e)}")
                    if not self.is_dry_run:
                        raise

            return run_urns

        except Exception as e:
            print(f"Error fetching runs from LangSmith: {str(e)}")
            if not self.is_dry_run:
                raise
            return []


def main():
    ingestion = LangSmithIngestion()
    print("\nStarting LangSmith metadata ingestion...")
    print(f"DataHub GMS URL: {os.getenv('DATAHUB_GMS_URL')}")

    # Get dry run mode from environment directly
    dry_run_value = os.getenv("DATAHUB_DRY_RUN", "false")
    is_dry_run = dry_run_value.lower() in ["true", "1", "yes", "on"]
    print(f"Mode: {'DRY RUN' if is_dry_run else 'LIVE'}")

    try:
        run_urns = ingestion.ingest_recent_runs()
        print(f"\nProcessed {len(run_urns)} runs")
        if is_dry_run:
            print("DRY RUN complete - metadata was printed but not sent to DataHub")
        else:
            print("Ingestion complete - metadata was sent to DataHub!")
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        if not is_dry_run:
            raise


if __name__ == "__main__":
    main()
