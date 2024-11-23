import os
from dotenv import load_dotenv
from langsmith import Client
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.com.linkedin.pegasus2avro.mxe import MetadataChangeEvent

load_dotenv()

class LangSmithIngestion:
    def __init__(self, gms_server="http://localhost:8080"):
        self.emitter = DatahubRestEmitter(gms_server)
        self.client = Client()

    def emit_run_metadata(self, run):
        """Emit metadata for a single LangSmith run"""
        run_urn = f"urn:li:llmRun:{run.id}"

        mce = MetadataChangeEvent(
            proposedSnapshot={
                "urn": run_urn,
                "aspects": [{
                    "llmRunProperties": {
                        "runId": run.id,
                        "startTime": str(run.start_time),
                        "endTime": str(run.end_time),
                        "status": run.status,
                        "inputs": run.inputs,
                        "outputs": run.outputs
                    }
                }]
            }
        )
        self.emitter.emit(mce)
        return run_urn

    def ingest_recent_runs(self, limit=100):
        """Ingest metadata from recent LangSmith runs"""
        runs = self.client.list_runs(limit=limit)
        run_urns = []

        for run in runs:
            try:
                run_urn = self.emit_run_metadata(run)
                run_urns.append(run_urn)
                print(f"Ingested run: {run.id}")
            except Exception as e:
                print(f"Error ingesting run {run.id}: {str(e)}")

        return run_urns

def main():
    ingestion = LangSmithIngestion()
    print("Starting LangSmith metadata ingestion...")
    run_urns = ingestion.ingest_recent_runs()
    print(f"Ingested {len(run_urns)} runs")
    print("Done!")

if __name__ == "__main__":
    main()
