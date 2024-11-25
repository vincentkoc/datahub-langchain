import json
from pathlib import Path

class JSONEmitter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, data: list, filename: str):
        file_path = self.output_dir / filename
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def emit_run(self, run):
        data = run.to_dict()
        filename = f"run_{run.id}.json"
        self.emit([data], filename)

    def emit_model(self, model):
        data = model.to_dict()
        filename = f"model_{model.name}.json"
        self.emit([data], filename)
