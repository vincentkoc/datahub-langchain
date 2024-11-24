import os
from datetime import datetime, timedelta
from uuid import UUID

from dotenv import load_dotenv
from langsmith import Client
from src.llm_metadata import LLMMetadataEmitter

load_dotenv()

class LangSmithIngestion:
    def __init__(self, gms_server: str = None):
        self.llm_emitter = LLMMetadataEmitter(gms_server)
        self.client = Client()
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "default")

    @property
    def is_dry_run(self) -> bool:
        return self.llm_emitter.is_dry_run

    def emit_run_metadata(self, run):
        """Emit metadata for a single LangSmith run"""
        try:
            print(f"\nDEBUG: Processing run {run.id}")

            # Add source tracking
            source_metadata = {
                "discovered_by": "langsmith",
                "run_type": getattr(run, "run_type", "unknown"),
                "parent_run_id": getattr(run, "parent_run_id", None),
                "project": self.project_name,
                "discovery_time": datetime.now().isoformat()
            }

            # Extract metadata
            token_usage = {}
            model_name = "unknown"
            if hasattr(run, "execution_metadata") and run.execution_metadata:
                token_usage = run.execution_metadata.get("token_usage", {})
                model_name = run.execution_metadata.get("model_name", "unknown")

            # First emit model metadata if available
            model_urn = None
            if model_name != "unknown":
                model_urn = self.llm_emitter.emit_model(
                    model_name=model_name,
                    provider="OpenAI",
                    model_type="chat",
                    capabilities=["text-generation", "chat"],
                    parameters={
                        "contextWindow": 4096,
                        "tokenLimit": 4096,
                        "costPerToken": 0.0001
                    },
                    source="langsmith"
                )

            # If this is a chain run, emit pipeline metadata
            pipeline_urn = None
            if getattr(run, "run_type", "") == "chain":
                pipeline_urn = self.llm_emitter.emit_pipeline(
                    chain_type=getattr(run, "name", "unknown"),
                    config={
                        "max_retries": 3,
                        "verbose": False,
                        "execution_metadata": run.execution_metadata if hasattr(run, "execution_metadata") else {},
                        "run_type": getattr(run, "run_type", "unknown"),
                        "serialized": getattr(run, "serialized", {}),
                        "extra": getattr(run, "extra", {}),
                        "source_metadata": source_metadata
                    },
                    upstream_urns=[model_urn] if model_urn else None,
                    source="langsmith"
                )

            # Emit run metadata
            run_urn = self.llm_emitter.emit_run(
                run_id=str(run.id),
                inputs=run.inputs,
                outputs=run.outputs,
                status=getattr(run, "status", "unknown"),
                metrics={
                    "tokenUsage": {
                        "promptTokens": token_usage.get("prompt_tokens", 0),
                        "completionTokens": token_usage.get("completion_tokens", 0),
                        "totalTokens": token_usage.get("total_tokens", 0)
                    },
                    "latency": getattr(run, "latency", 0),
                    "cost": getattr(run, "cost", 0)
                },
                upstream_urns=[pipeline_urn] if pipeline_urn else ([model_urn] if model_urn else None),
                error=str(run.error) if hasattr(run, "error") and run.error else None,
                tags=list(getattr(run, "tags", [])),
                feedback_stats=getattr(run, "feedback_stats", {})
            )

            # Emit lineage
            lineage_pairs = []

            # Model -> Pipeline lineage
            if model_urn and pipeline_urn:
                lineage_pairs.append((pipeline_urn, model_urn, "Uses"))

            # Pipeline -> Run lineage
            if pipeline_urn and run_urn:
                lineage_pairs.append((run_urn, pipeline_urn, "ExecutedBy"))

            # Direct Model -> Run lineage if no pipeline
            elif model_urn and run_urn:
                lineage_pairs.append((run_urn, model_urn, "Uses"))

            # Emit all lineage relationships
            for source, target, lineage_type in lineage_pairs:
                try:
                    self.llm_emitter.emit_lineage(source, target, lineage_type)
                except Exception as e:
                    print(f"Failed to emit lineage {source} -> {target}: {e}")
                    if not self.is_dry_run:
                        raise

            return run_urn

        except Exception as e:
            print(f"Error processing run {run.id}: {str(e)}")
            if not self.is_dry_run:
                raise
            return None

    def ingest_recent_runs(self, limit=50, days_ago=7):
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
