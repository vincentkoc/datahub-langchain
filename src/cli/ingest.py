import click
from datetime import datetime, timedelta
from typing import Optional

from ..config import ObservabilityConfig, ObservabilitySetup
from ..collectors.run_collector import RunCollector
from ..collectors.model_collector import ModelCollector

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
def ingest(days: int, limit: int, platform: str, env_file: Optional[str], batch_size: int):
    """Ingest historical data from observability platforms"""

    # Load configuration
    config = ObservabilityConfig.from_env(env_file)

    # Override config with CLI parameters
    config.ingest_window_days = days
    config.ingest_limit = limit
    config.ingest_batch_size = batch_size

    # Initialize observability
    obs = ObservabilitySetup(config)
    obs.setup()

    # Get connector for specified platform
    connector = obs.get_connector(platform)
    if not connector:
        click.echo(f"Error: Platform {platform} not configured")
        return

    # Setup collectors
    run_collector = RunCollector([connector])
    model_collector = ModelCollector([connector])

    # Get emitter
    emitter = obs.get_emitter()

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
        for i in range(0, len(runs), config.ingest_batch_size):
            batch = runs[i:i + config.ingest_batch_size]
            click.echo(f"\nProcessing batch {i//config.ingest_batch_size + 1}...")

            for run in batch:
                try:
                    emitter.emit_run(run)
                    click.echo(f"✓ Emitted run: {run.id}")
                except Exception as e:
                    click.echo(f"✗ Failed to emit run {run.id}: {e}")

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
