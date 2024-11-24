#!/usr/bin/env python
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from dotenv import load_dotenv
from src.cli.ingest import ingest

# Find and load the nearest .env file
env_file = root_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)

if __name__ == "__main__":
    # Run ingestion with default parameters
    ingest.callback(
        days=7,
        limit=1000,
        platform="langsmith",
        env_file=str(env_file) if env_file.exists() else None,
        batch_size=100
    )
