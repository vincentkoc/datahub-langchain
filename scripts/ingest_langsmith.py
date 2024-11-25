#!/usr/bin/env python
import os
import sys
from pathlib import Path
import argparse
import json
import logging
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add the src directory to the Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from dotenv import load_dotenv
from src.cli.ingest import ingest

# Find the .env file by traversing up the directories
env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)
    logging.info(f"Loaded environment variables from {env_file}")
else:
    logging.warning(".env file not found. Environment variables may be missing.")

# Access API key
API_KEY = os.getenv('LANGSMITH_API_KEY')
if not API_KEY:
    logging.error("LANGSMITH_API_KEY not found in environment variables.")

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

    args = parser.parse_args()

    # Prepare the processing directory
    if args.save_debug_data:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processing_dir = root_dir / 'metadata' / 'processing' / timestamp
        processing_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Debug data will be saved to {processing_dir}")
    else:
        processing_dir = None

    # Run ingestion and capture data
    raw_data, processed_data = ingest.callback(
        days=args.days,
        limit=args.limit,
        platform=args.platform,
        env_file=args.env_file if args.env_file else (str(env_file) if env_file else None),
        batch_size=args.batch_size,
        debug=args.debug,
        hard_fail=args.hard_fail,
        save_debug_data=args.save_debug_data,
        processing_dir=processing_dir  # Pass the directory to save data
    )

    # No need to save data here since it's handled inside the callback

if __name__ == "__main__":
    main()
