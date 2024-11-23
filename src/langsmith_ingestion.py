import os
from dotenv import load_dotenv
from langsmith import Client
from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent
from metadata_setup import get_datahub_emitter

load_dotenv()

class LangSmithIngestion:
    def __init__(self, gms_server: str = None):
        self.emitter = get_datahub_emitter(gms_server)
        self.client = Client()
        self.is_dry_run = os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true"

    def emit_metadata(self, mce: MetadataChangeEvent) -> str:
        """Emit metadata with dry run handling"""
        try:
            if self.is_dry_run:
                print(f"\nDRY RUN: Would emit metadata for {mce['proposedSnapshot']['urn']}")
                self.emitter.emit(mce)  # DryRunEmitter will handle printing
            else:
                self.emitter.emit(mce)
            return mce['proposedSnapshot']['urn']
        except Exception as e:
            error_msg = f"Failed to emit metadata: {str(e)}"
            if self.is_dry_run:
                print(f"DRY RUN ERROR: {error_msg}")
            else:
                raise Exception(error_msg)

    def emit_run_metadata(self, run):
        """Emit metadata for a single LangSmith run"""
        run_urn = f"urn:li:llmRun:{run.id}"

        # Extract token usage if available
        token_usage = {}
        if hasattr(run, 'execution_metadata') and run.execution_metadata:
            token_usage = run.execution_metadata.get('token_usage', {})

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

        mce = MetadataChangeEvent(
            proposedSnapshot={
                "urn": run_urn,
                "aspects": [{
                    "llmRunProperties": {
                        "runId": run.id,
                        "name": getattr(run, 'name', None),
                        "startTime": str(run.start_time),
                        "endTime": str(run.end_time),
                        "status": run.status,
                        "inputs": run.inputs,
                        "outputs": run.outputs,
                        "error": getattr(run, 'error', None),
                        "runtime": getattr(run, 'runtime_seconds', None),
                        "parentRunId": getattr(run, 'parent_run_id', None),
                        "childRunIds": getattr(run, 'child_run_ids', []),
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
                }]
            }
        )
        return self.emit_metadata(mce)

    def ingest_recent_runs(self, limit=100):
        """Ingest metadata from recent LangSmith runs"""
        if self.is_dry_run:
            print(f"\nDRY RUN: Fetching up to {limit} recent runs from LangSmith")

        # Always fetch runs from LangSmith
        runs = self.client.list_runs(limit=limit)
        run_urns = []

        for run in runs:
            try:
                run_urn = self.emit_run_metadata(run)
                run_urns.append(run_urn)
                if self.is_dry_run:
                    print(f"DRY RUN: Processing run: {run.id}")
                else:
                    print(f"Ingested run: {run.id}")
            except Exception as e:
                print(f"Error processing run {run.id}: {str(e)}")
                if not self.is_dry_run:
                    raise

        return run_urns

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
        if not is_dry_run:
            raise

if __name__ == "__main__":
    main()
