import os
import json
from datetime import datetime, timedelta
from uuid import UUID
from dotenv import load_dotenv
from langsmith import Client
from metadata_setup import get_datahub_emitter

load_dotenv()

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
        self.is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "default")

    def serialize_run_data(self, data):
        """Serialize run data, converting UUIDs to strings"""
        return json.loads(json.dumps(data, cls=JSONEncoder))

    def emit_metadata(self, mce_dict: dict) -> str:
        """Emit metadata with dry run handling"""
        try:
            if self.is_dry_run:
                print(f"\nDRY RUN: Would emit metadata for {mce_dict['proposedSnapshot']['urn']}")
            self.emitter.emit(mce_dict)
            return mce_dict['proposedSnapshot']['urn']
        except Exception as e:
            error_msg = f"Failed to emit metadata: {str(e)}"
            if self.is_dry_run:
                print(f"DRY RUN ERROR: {error_msg}")
            else:
                raise Exception(error_msg)
            return None

    def emit_run_metadata(self, run):
        """Emit metadata for a single LangSmith run"""
        run_urn = f"urn:li:llmRun:{run.id}"

        # Clean up error message if present
        error_message = None
        if hasattr(run, 'error') and run.error:
            # Extract just the main error message without the full traceback
            error_str = str(run.error)
            error_message = error_str.split('\n')[0] if '\n' in error_str else error_str

        # Extract token usage if available
        token_usage = {}
        if hasattr(run, 'execution_metadata') and run.execution_metadata:
            token_usage = self.serialize_run_data(run.execution_metadata.get('token_usage', {}))

        # Get feedback if available
        feedback_list = []
        if hasattr(run, 'feedback_list'):
            feedback_list = [
                {
                    "key": fb.key,
                    "value": fb.value,
                    "comment": fb.comment,
                    "timestamp": str(fb.timestamp)
                }
                for fb in run.feedback_list
            ]

        # Serialize all run data
        run_data = {
            "runId": str(run.id),
            "name": getattr(run, 'name', None),
            "startTime": str(run.start_time),
            "endTime": str(run.end_time),
            "status": run.status,
            "inputs": self.serialize_run_data(run.inputs),
            "outputs": self.serialize_run_data(run.outputs),
            "error": error_message,  # Using cleaned error message
            "runtime": getattr(run, 'runtime_seconds', None),
            "parentRunId": str(run.parent_run_id) if getattr(run, 'parent_run_id', None) else None,
            "childRunIds": [str(id) for id in getattr(run, 'child_run_ids', [])],
            "tags": getattr(run, 'tags', []),
            "feedback": feedback_list,
            "metrics": {
                "tokenUsage": {
                    "promptTokens": token_usage.get('prompt_tokens'),
                    "completionTokens": token_usage.get('completion_tokens'),
                    "totalTokens": token_usage.get('total_tokens')
                },
                "latency": getattr(run, 'latency', None),
                "cost": getattr(run, 'cost', None)
            },
            "trace": {
                "projectName": getattr(run, 'project_name', None),
                "sessionName": getattr(run, 'session_name', None),
                "executionOrder": getattr(run, 'execution_order', None)
            }
        }

        mce_dict = {
            "proposedSnapshot": {
                "urn": run_urn,
                "aspects": [{
                    "llmRunProperties": run_data
                }]
            }
        }
        return self.emit_metadata(mce_dict)

    def ingest_recent_runs(self, limit=100, days_ago=7):
        """Ingest metadata from recent LangSmith runs"""
        if self.is_dry_run:
            print(f"\nDRY RUN: Fetching up to {limit} recent runs from LangSmith")

        # Get runs from the last X days
        start_time = datetime.now() - timedelta(days=days_ago)

        try:
            # List runs with proper filters
            runs = self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                execution_order=1,  # Get top-level runs
                limit=limit
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
    is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"
    if is_dry_run:
        print("\nRunning in DRY RUN mode - metadata will be printed but not sent to DataHub")

    ingestion = LangSmithIngestion()
    print("Starting LangSmith metadata ingestion...")

    try:
        run_urns = ingestion.ingest_recent_runs()
        print(f"\nProcessed {len(run_urns)} runs")
        if is_dry_run:
            print("DRY RUN complete - metadata was printed but not sent to DataHub")
        else:
            print("Ingestion complete!")
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        if not self.is_dry_run:
            raise

if __name__ == "__main__":
    main()
