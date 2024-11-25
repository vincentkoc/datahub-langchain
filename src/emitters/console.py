from typing import Dict, Any
import json
from datetime import datetime

from ..base import LLMMetadataEmitter, LLMModel, LLMRun, LLMChain

class ConsoleEmitter(LLMMetadataEmitter):
    """Emits LLM metadata to console for debugging"""

    def __init__(self, pretty_print: bool = True):
        self.pretty_print = pretty_print

    def _print_json(self, data: Dict[str, Any]) -> None:
        if self.pretty_print:
            print(json.dumps(data, indent=2, default=str))
        else:
            print(json.dumps(data, default=str))

    def emit_model(self, model: LLMModel) -> str:
        """Print model metadata to console"""
        print("\n=== Model Metadata ===")
        self._print_json({
            "name": model.name,
            "provider": model.provider,
            "model_family": model.model_family,
            "capabilities": model.capabilities,
            "parameters": model.parameters,
            "metadata": model.metadata
        })
        return f"model:{model.provider}/{model.name}"

    def emit_run(self, run: LLMRun) -> str:
        """Print run metadata to console"""
        print("\n=== Run Metadata ===")
        self._print_json({
            "id": run.id,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "model": run.model.name if run.model else None,
            "inputs": run.inputs,
            "outputs": run.outputs,
            "metrics": run.metrics,
            "parent_id": run.parent_id,
            "metadata": run.metadata
        })
        return f"run:{run.id}"

    def emit_chain(self, chain: LLMChain) -> str:
        """Print chain metadata to console"""
        print("\n=== Chain Metadata ===")
        self._print_json({
            "id": chain.id,
            "name": chain.name,
            "components": chain.components,
            "config": chain.config,
            "metadata": chain.metadata
        })
        return f"chain:{chain.id}"

    def emit_lineage(self, source_urn: str, target_urn: str, lineage_type: str) -> None:
        """Print lineage relationship to console"""
        print("\n=== Lineage ===")
        self._print_json({
            "source": source_urn,
            "target": target_urn,
            "type": lineage_type
        })
