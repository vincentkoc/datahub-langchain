import json
import os
import time
from datetime import datetime, timedelta
from uuid import UUID

from dotenv import load_dotenv
from langsmith import Client
from datahub.emitter.mce_builder import make_dataset_urn
from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    StatusClass,
    BrowsePathsClass,
    DatasetSnapshotClass,
    MetadataChangeEventClass,
)

from src.metadata_setup import get_datahub_emitter, DryRunEmitter

load_dotenv()

# Increase delay between metadata pushes (in seconds)
METADATA_PUSH_DELAY = 2.0


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle UUID objects"""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


class LangSmithIngestion:
    def __init__(self, gms_server: str = None):
        self.emitter = get_datahub_emitter(gms_server)
        self.client = Client()
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "default")

    @property
    def is_dry_run(self) -> bool:
        return isinstance(self.emitter, DryRunEmitter)

    def emit_run_metadata(self, run):
        """Emit metadata for a single LangSmith run using Dataset entity"""
        try:
            print(f"\nDEBUG: Processing run {run.id}")

            # Extract metadata
            error_message = None
            if hasattr(run, "error") and run.error:
                error_str = str(run.error)
                error_message = error_str.split("\n")[0] if "\n" in error_str else error_str

            token_usage = {}
            model_name = "unknown"
            if hasattr(run, "execution_metadata") and run.execution_metadata:
                token_usage = run.execution_metadata.get("token_usage", {})
                model_name = run.execution_metadata.get("model_name", "unknown")

            # Create Dataset URN
            dataset_urn = make_dataset_urn(
                platform="langsmith",
                name=f"run_{run.id}",
                env="PROD"
            )

            # Create browse paths
            browse_paths = BrowsePathsClass(
                paths=[
                    "/langsmith/runs",
                    f"/langsmith/runs/{run.id}",
                    f"/langsmith/models/{model_name}/runs"
                ]
            )

            # Create Dataset properties with metadata
            properties = DatasetPropertiesClass(
                name=str(run.id),
                description=f"LangSmith Run {run.id}",
                customProperties={
                    "runId": str(run.id),
                    "name": str(getattr(run, "name", "")),
                    "startTime": str(getattr(run, "start_time", "")),
                    "endTime": str(getattr(run, "end_time", "")),
                    "status": str(getattr(run, "status", "unknown")),
                    "inputs": json.dumps(dict(getattr(run, "inputs", {})), cls=JSONEncoder),
                    "outputs": json.dumps(dict(getattr(run, "outputs", {})), cls=JSONEncoder) if getattr(run, "outputs", None) else "",
                    "error": error_message if error_message else "",
                    "runtime": str(float(getattr(run, "runtime_seconds", 0))),
                    "parentRunId": str(getattr(run, "parent_run_id", "")) if getattr(run, "parent_run_id", None) else "",
                    "childRunIds": json.dumps([str(id) for id in getattr(run, "child_run_ids", [])]),
                    "tags": json.dumps(list(getattr(run, "tags", []))),
                    "tokenUsage": json.dumps({
                        "promptTokens": token_usage.get("prompt_tokens", 0),
                        "completionTokens": token_usage.get("completion_tokens", 0),
                        "totalTokens": token_usage.get("total_tokens", 0),
                    }),
                    "latency": str(float(getattr(run, "latency", 0))),
                    "cost": str(float(getattr(run, "cost", 0))),
                    "modelName": model_name
                }
            )

            # Create Status aspect
            status = StatusClass(removed=False)

            # Create MetadataChangeEvent
            mce = MetadataChangeEventClass(
                proposedSnapshot=DatasetSnapshotClass(
                    urn=dataset_urn,
                    aspects=[
                        status,
                        properties,
                        browse_paths
                    ]
                )
            )

            # Debug output
            print(f"\nEmitting metadata for run {run.id}")
            print(f"URN: {dataset_urn}")
            print("Metadata structure:")
            print(json.dumps(mce.to_obj(), indent=2, cls=JSONEncoder))

            # Emit metadata
            self.emitter.emit(mce)
            print(f"Successfully emitted metadata for run {run.id}")

            return dataset_urn

        except Exception as e:
            print(f"Error processing run {run.id}: {str(e)}")
            if not self.is_dry_run:
                raise
            return None

    def ingest_recent_runs(self, limit=100, days_ago=7):
        """Ingest metadata from recent LangSmith runs"""
        start_time = datetime.now() - timedelta(days=days_ago)

        try:
            print(f"\nFetching runs from the last {days_ago} days (limit: {limit})...")
            runs = self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                execution_order=1,
                limit=limit,
            )

            run_urns = []
            for run in runs:
                try:
                    run_urn = self.emit_run_metadata(run)
                    if run_urn:
                        run_urns.append(run_urn)
                        print(f"Successfully processed run: {run.id}")
                        if not self.is_dry_run:
                            time.sleep(METADATA_PUSH_DELAY)
                except Exception as e:
                    print(f"Error processing run {run.id}: {str(e)}")
                    if not self.is_dry_run:
                        raise

            print(f"\nProcessed {len(run_urns)} runs")
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
    print(f"Mode: {'DRY RUN' if ingestion.is_dry_run else 'LIVE'}")

    try:
        run_urns = ingestion.ingest_recent_runs()
        if ingestion.is_dry_run:
            print("\nDRY RUN complete - metadata was printed but not sent to DataHub")
        else:
            print("\nIngestion complete - metadata was sent to DataHub!")
            print(f"Processed {len(run_urns)} runs")
    except Exception as e:
        print(f"\nError during ingestion: {str(e)}")
        if not ingestion.is_dry_run:
            raise


if __name__ == "__main__":
    main()
