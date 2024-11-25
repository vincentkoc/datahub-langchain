import inspect
import os
from pathlib import Path

def detect_pipeline_name(default: str = "default_pipeline") -> str:
    """Detect pipeline name from calling file"""
    # Get the calling frame (skip current frame and library internals)
    frame = inspect.currentframe()
    while frame:
        filename = frame.f_code.co_filename
        if not any(x in filename for x in ['langchain', 'datahub_langchain', 'pipeline_utils.py']):
            break
        frame = frame.f_back

    if frame:
        # Get filename without extension and path
        pipeline = os.path.splitext(os.path.basename(filename))[0]
        return f"{pipeline}-pipeline"
    return default
