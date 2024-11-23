import json
import os
import time
from datetime import datetime, timedelta
from uuid import UUID

from dotenv import load_dotenv
from langsmith import Client
from datahub.emitter.mce_builder import make_dataset_urn
from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent
from datahub.metadata.schema_classes import DatasetSnapshotClass, StatusClass, DatasetPropertiesClass

from src.metadata_setup import get_datahub_emitter, DryRunEmitter

load_dotenv()

# Add delay between metadata pushes (in seconds)
METADATA_PUSH_DELAY = 1.0


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

            # Extract token usage and model name safely
            token_usage = {}
            model_name = "unknown"
            if hasattr(run, "execution_metadata") and run.execution_metadata:
                token_usage = run.execution_metadata.get("token_usage", {})
                model_name = run.execution_metadata.get("model_name", "unknown")

            # Create dataset URN
            run_urn = make_dataset_urn(
                platform="llm",
                name=f"run_{run.id}",
                env="PROD"
            )

            # Create Status aspect
            status = StatusClass(removed=False)

            # Add browse paths
            browse_paths = [
                "/llm/runs",
                f"/llm/runs/{run.id}",
                f"/llm/models/{model_name}/runs"
            ]

            # Add relationships
            relationships = []
            if hasattr(run, "model_id") and run.model_id:
                relationships.append({
                    "type": "RunsOn",
                    "target": f"urn:li:dataset:(urn:li:dataPlatform:datahub,llmModel,{run.model_id})"
                })

            # Create Properties aspect with only string values
            custom_properties = {
                "runId": str(run.id),
                "name": str(getattr(run, "name", "")),
                "startTime": str(getattr(run, "start_time", "")),
                "endTime": str(getattr(run, "end_time", "")),
                "status": str(getattr(run, "status", "unknown")),
                "inputs": json.dumps(dict(getattr(run, "inputs", {}))),
                "outputs": json.dumps(dict(getattr(run, "outputs", {}))) if getattr(run, "outputs", None) else "",
                "error": error_message if error_message else "",
                "runtime": str(float(getattr(run, "runtime_seconds", 0))),
                "parentRunId": str(getattr(run, "parent_run_id", "")) if getattr(run, "parent_run_id", None) else "",
                "childRunIds": json.dumps([str(id) for id in getattr(run, "child_run_ids", [])]),
                "tags": json.dumps(list(getattr(run, "tags", []))),
                "feedback": json.dumps([]),
                "tokenUsage": json.dumps({
                    "promptTokens": token_usage.get("prompt_tokens", 0),
                    "completionTokens": token_usage.get("completion_tokens", 0),
                    "totalTokens": token_usage.get("total_tokens", 0),
                }),
                "latency": str(float(getattr(run, "latency", 0))),
                "cost": str(float(getattr(run, "cost", 0))),
                "browsePaths": json.dumps(browse_paths),
                "relationships": json.dumps(relationships),
                "modelName": model_name
            }

            properties = DatasetPropertiesClass(
                description=f"LangSmith Run {run.id}",
                name=str(run.id),
                customProperties=custom_properties
            )

            # Create DatasetSnapshot
            snapshot = DatasetSnapshotClass(
                urn=run_urn,
                aspects=[status, properties]
            )

            # Create MetadataChangeEvent
            mce = MetadataChangeEvent(
                proposedSnapshot=snapshot
            )

            # Emit and return URN
            self.emitter.emit(mce)
            return run_urn

        except Exception as e:
            print(f"Error processing run {run.id}: {str(e)}")
            if not self.is_dry_run:
                raise  # Re-raise exception in live mode
            return None

    def verify_run_metadata(self, run_urn):
        """Verify that run metadata was properly written to DataHub"""
        try:
            # Get server URL from environment
            server_url = os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")

            # Create GraphQL query with proper subselection
            query = """
            query getDataset($urn: String!) {
                dataset(urn: $urn) {
                    urn
                    properties {
                        description
                        customProperties {
                            key
                            value
                        }
                    }
                }
            }
            """

            # Make GraphQL request
            response = self.emitter._session.post(
                f"{server_url}/api/graphql",
                json={
                    "query": query,
                    "variables": {
                        "urn": run_urn
                    }
                }
            )

            print(f"\nVerifying run metadata for URN: {run_urn}")
            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if data and data.get('data', {}).get('dataset'):
                    print("✓ Run metadata found")
                    dataset = data['data']['dataset']
                    properties = dataset.get('properties', {})

                    # Pretty print custom properties
                    custom_props = {}
                    for prop in properties.get('customProperties', []):
                        key = prop.get('key')
                        value = prop.get('value')
                        if key and value:
                            custom_props[key] = value

                    if custom_props:
                        print("\nRun Properties:")
                        for key, value in custom_props.items():
                            if key in ['inputs', 'outputs', 'tokenUsage', 'browsePaths', 'relationships']:
                                try:
                                    parsed = json.loads(value)
                                    print(f"  {key}: {json.dumps(parsed, indent=2)}")
                                except:
                                    print(f"  {key}: {value}")
                            else:
                                print(f"  {key}: {value}")
                    return True
                else:
                    print(f"✗ Run metadata not found for URN: {run_urn}")
                    return False
            else:
                print(f"✗ Error verifying run metadata")
                print(f"  Status code: {response.status_code}")
                print(f"  Response: {response.text[:500]}")
                return False

        except Exception as e:
            print(f"✗ Error during verification: {str(e)}")
            print(f"  Full error: {repr(e)}")
            if not self.is_dry_run:
                raise
            return False

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
                            # Verify the metadata was written
                            if self.verify_run_metadata(run_urn):
                                print(f"✓ Verified metadata for run: {run.id}")
                            else:
                                print(f"✗ Failed to verify metadata for run: {run.id}")
                            # Add delay between metadata pushes
                            time.sleep(METADATA_PUSH_DELAY)
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
