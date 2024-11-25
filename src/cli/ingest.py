import sys
import click
from datetime import datetime, timedelta
from typing import Optional, Tuple

from ..config import ObservabilityConfig, ObservabilitySetup
from ..collectors.run_collector import RunCollector
from ..collectors.model_collector import ModelCollector
from ..emitters.datahub import DataHubEmitter
from ..platforms.langsmith import LangsmithIngestor

@click.group()
def cli():
    """LLM Observability CLI"""
    pass

@cli.command()
@click.option('--days', default=7, help='Number of days of history to ingest')
@click.option('--limit', default=1000, help='Maximum number of records to ingest')
@click.option('--platform', default='langsmith', help='Platform to ingest from')
@click.option('--env-file', default=None, help='Path to .env file')
@click.option('--batch-size', default=100, help='Batch size for ingestion')
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
@click.option('--hard-fail/--no-hard-fail', default=True, help='Stop on first failure')
@click.option('--save-debug-data/--no-save-debug-data', default=True, help='Save raw and processed data for debugging')
@click.option('--processing-dir', default=None, help='Path to the processing directory')
@click.option('--emit-to-datahub/--no-emit-to-datahub', default=True, help='Emit data to DataHub')
def ingest(days: int, limit: int, platform: str, env_file: Optional[str],
           batch_size: int, debug: bool, hard_fail: bool, save_debug_data: bool,
           processing_dir: Optional[str], emit_to_datahub: bool):
    """Ingest historical data from observability platforms"""
    ingest_logic(days, limit, platform, env_file, batch_size, debug,
                 hard_fail, save_debug_data, processing_dir, emit_to_datahub)

def ingest_logic(days: int, limit: int, platform: str, env_file: Optional[str],
                 batch_size: int, debug: bool, hard_fail: bool, save_debug_data: bool,
                 processing_dir: Optional[str], emit_to_datahub: bool = True) -> Tuple[list, list]:
    """Ingest historical data from observability platforms"""

    # Load configuration
    config = ObservabilityConfig.from_env(env_file)

    # Initialize observability with debug and hard fail settings
    obs = ObservabilitySetup(config)

    # Get emitter with debug and hard fail settings
    datahub_emitter = DataHubEmitter(
        gms_server=config.datahub_gms_url,
        debug=debug,
        hard_fail=hard_fail
    )

    if debug:
        print("\nRunning in debug mode")
        print(f"Hard fail mode: {'enabled' if hard_fail else 'disabled'}")

    # Override config with CLI parameters
    config.ingest_window_days = days
    config.ingest_limit = limit
    config.ingest_batch_size = batch_size

    # Initialize observability
    obs.setup()

    # Get connector for specified platform
    connector = obs.get_connector(platform)
    if not connector:
        print(f"Error: Platform {platform} not configured")
        return [], []

    # Setup collectors
    run_collector = RunCollector([connector])
    model_collector = ModelCollector([connector])

    print(f"\nIngesting data from {platform}...")
    print(f"Time window: {config.ingest_window_days} days")
    print(f"Batch size: {config.ingest_batch_size}")

    try:
        # Collect models
        print("\nCollecting models...")
        models = model_collector.collect_models()
        print(f"Found {len(models)} models")

        # Collect runs
        print("\nCollecting runs...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=config.ingest_window_days)

        runs = run_collector.collect_runs(
            start_time=start_time,
            end_time=end_time,
            limit=config.ingest_limit
        )
        print(f"Found {len(runs)} runs")

        # Initialize the ingestor
        ingestor = LangsmithIngestor(
            config,
            save_debug_data=save_debug_data,
            processing_dir=processing_dir,
            emit_to_datahub=emit_to_datahub,
            datahub_emitter=datahub_emitter if emit_to_datahub else None
        )

        # Process and emit models
        print("\nProcessing and emitting models...")
        ingestor.process_models(models)
        ingestor.emit_models(models)

        # Process and emit runs
        print("\nProcessing and emitting runs...")
        processed_data = ingestor.process_data(runs)
        ingestor.emit_data(processed_data)

        # Print statistics
        print("\nRun Statistics:")
        stats = run_collector.get_run_stats(timedelta(days=config.ingest_window_days))
        for key, value in stats.items():
            print(f"{key}: {value}")

        print("\nModel Statistics:")
        stats = model_collector.get_model_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

        # Return raw and processed data
        return runs, processed_data

    except Exception as e:
        print(f"\nError during ingestion: {str(e)}")
        if not config.datahub_dry_run:
            raise

        # Ensure we return empty lists in case of exception
        return [], []

# Modify the ingest function to be callable
def ingest_callback(**kwargs):
    return ingest.invoke(ingest.make_context('ingest', list(kwargs.items())))

# Update scripts/ingest_langsmith.py to use ingest_callback

if __name__ == '__main__':
    cli()
