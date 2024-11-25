#!/usr/bin/env python
import os
import sys
from pathlib import Path
import argparse
import logging
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add the src directory to the Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.cli.ingest import ingest_logic

# Find the .env file by traversing up the directories
env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)
    logging.info(f"Loaded environment variables from {env_file}")
else:
    logging.warning(".env file not found. Environment variables may be missing.")

def main():
    parser = argparse.ArgumentParser(description='Ingest data from Langsmith.')
    parser.add_argument('--days', type=int, default=7, help='Number of days to ingest')
    parser.add_argument('--limit', type=int, default=1000, help='Maximum number of runs to ingest')
    parser.add_argument('--platform', default='langsmith', help='Platform to ingest from')
    parser.add_argument('--env-file', help='Path to the environment file')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for ingestion')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--hard-fail', action='store_true', help='Enable hard fail mode')
    parser.add_argument('--save-debug-data', type=bool, nargs='?', const=True, default=True,
                        help='Save raw and processed data for debugging (default: True)')
    parser.add_argument('--emit-to-datahub', action='store_true', default=True,
                        help='Emit data to DataHub (default: True)')

    args = parser.parse_args()

    # Prepare the processing directory
    if args.save_debug_data:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processing_dir = root_dir / 'metadata' / 'processing' / args.platform / timestamp
        processing_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Debug data will be saved to {processing_dir}")
    else:
        processing_dir = None

    # Run ingestion and capture data
    raw_data, processed_data = ingest_logic(
        days=args.days,
        limit=args.limit,
        platform=args.platform,
        env_file=args.env_file if args.env_file else (str(env_file) if env_file else None),
        batch_size=args.batch_size,
        debug=args.debug,
        hard_fail=args.hard_fail,
        save_debug_data=args.save_debug_data,
        processing_dir=processing_dir,
        emit_to_datahub=args.emit_to_datahub
    )

if __name__ == "__main__":
    main()
