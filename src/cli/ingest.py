import sys
import click
from datetime import datetime, timedelta
from typing import Optional

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
def ingest(days: int, limit: int, platform: str, env_file: Optional[str],
          batch_size: int, debug: bool, hard_fail: bool, save_debug_data: bool, processing_dir: Optional[str]):
    """Ingest historical data from observability platforms"""

    # Load configuration
    config = ObservabilityConfig.from_env(env_file)

    # Initialize observability with debug and hard fail settings
    obs = ObservabilitySetup(config)

    # Get emitter with debug and hard fail settings
    emitter = DataHubEmitter(
        gms_server=config.datahub_gms_url,
        debug=debug,
        hard_fail=hard_fail
    )

    if debug:
        click.echo("\nRunning in debug mode")
        click.echo(f"Hard fail mode: {'enabled' if hard_fail else 'disabled'}")

    # Override config with CLI parameters
    config.ingest_window_days = days
    config.ingest_limit = limit
    config.ingest_batch_size = batch_size

    # Initialize observability
    obs.setup()

    # Get connector for specified platform
    connector = obs.get_connector(platform)
    if not connector:
        click.echo(f"Error: Platform {platform} not configured")
        return

    # Setup collectors
    run_collector = RunCollector([connector])
    model_collector = ModelCollector([connector])

    click.echo(f"\nIngesting data from {platform}...")
    click.echo(f"Time window: {config.ingest_window_days} days")
    click.echo(f"Batch size: {config.ingest_batch_size}")

    try:
        # Collect and emit models
        click.echo("\nCollecting models...")
        models = model_collector.collect_models()
        click.echo(f"Found {len(models)} models")

        for model in models:
            try:
                emitter.emit_model(model)
                if debug:
                    click.echo(f"✓ Emitted model: {model.name}")
            except Exception as e:
                click.echo(f"✗ Failed to emit model {model.name}: {e}")

        # Collect and emit runs
        click.echo("\nCollecting runs...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=config.ingest_window_days)

        runs = run_collector.collect_runs(
            start_time=start_time,
            end_time=end_time,
            limit=config.ingest_limit
        )
        click.echo(f"Found {len(runs)} runs")

        # Process runs in batches
        for i in range(0, len(runs), batch_size):
            batch = runs[i:i + batch_size]
            click.echo(f"\nProcessing batch {i//batch_size + 1}...")

            for run in batch:
                try:
                    emitter.emit_run(run)
                    if debug:
                        click.echo(f"✓ Emitted run: {run.id}")
                except Exception as e:
                    click.echo(f"✗ Failed to emit run {run.id}: {e}")
                    if hard_fail:
                        click.echo("\nStopping due to hard fail mode")
                        sys.exit(1)  # Exit immediately on failure

        # Print statistics
        click.echo("\nRun Statistics:")
        stats = run_collector.get_run_stats(timedelta(days=config.ingest_window_days))
        for key, value in stats.items():
            click.echo(f"{key}: {value}")

        click.echo("\nModel Statistics:")
        stats = model_collector.get_model_stats()
        for key, value in stats.items():
            click.echo(f"{key}: {value}")

    except Exception as e:
        click.echo(f"\nError during ingestion: {str(e)}")
        if not config.datahub_dry_run:
            raise

if __name__ == '__main__':
    cli()
