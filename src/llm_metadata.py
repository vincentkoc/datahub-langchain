from typing import Dict, List, Optional
from datetime import datetime

from .metadata_emitter import BaseMetadataEmitter

class LLMMetadataEmitter:
    """Metadata emitter for LLM components"""

    def __init__(self, gms_server: str = None):
        self.model_emitter = BaseMetadataEmitter("llm", gms_server)
        self.prompt_emitter = BaseMetadataEmitter("llm", gms_server)
        self.chain_emitter = BaseMetadataEmitter("llm", gms_server)
        self.run_emitter = BaseMetadataEmitter("llm", gms_server)

    @property
    def is_dry_run(self) -> bool:
        return self.model_emitter.is_dry_run

    def emit_model(self, model_name: str, provider: str, model_type: str,
                  capabilities: List[str], parameters: Dict, **kwargs) -> str:
        """Emit LLM model metadata"""
        return self.model_emitter.emit_metadata(
            name=model_name,
            metadata={
                "provider": provider,
                "model_type": model_type,
                "capabilities": capabilities,
                "parameters": parameters
            },
            description=f"LLM Model: {model_name}",
            browse_paths=[
                "/llm/models",
                f"/llm/models/{provider.lower()}",
                f"/llm/models/{provider.lower()}/{model_name}"
            ],
            entity_type="MLMODEL",
            sub_type="model",
            icon_url="https://raw.githubusercontent.com/datahub-project/datahub/master/datahub-web-react/src/images/openai.png",
            **kwargs
        )

    def emit_prompt(self, prompt, template_format: str, category: str,
                   version: str, **kwargs) -> str:
        """Emit prompt metadata"""
        formatted_messages = self._format_messages(prompt.messages)

        return self.prompt_emitter.emit_metadata(
            name=f"prompt_{hash(str(formatted_messages))}",
            metadata={
                "template": formatted_messages,
                "format": template_format,
                "category": category,
                "version": version,
                "input_variables": list(prompt.input_variables)
            },
            description="Chat prompt template",
            browse_paths=[
                "/llm/prompts",
                "/llm/prompts/chat",
                f"/llm/prompts/chat/{category.lower()}"
            ],
            entity_type="DATASET",
            sub_type="prompt",
            icon_url="https://raw.githubusercontent.com/datahub-project/datahub/master/datahub-web-react/src/images/langchain.png",
            **kwargs
        )

    def emit_chain(self, chain_type: str, config: Dict,
                  upstream_urns: List[str], **kwargs) -> str:
        """Emit chain metadata"""
        return self.chain_emitter.emit_metadata(
            name=f"chain_{chain_type}",
            metadata={
                "type": chain_type,
                "config": config
            },
            description=f"LLM Chain: {chain_type}",
            browse_paths=[
                "/llm/chains",
                f"/llm/chains/{chain_type.lower()}"
            ],
            entity_type="DATAFLOW",
            sub_type="chain",
            upstream_urns=upstream_urns,
            icon_url="https://raw.githubusercontent.com/datahub-project/datahub/master/datahub-web-react/src/images/langchain.png",
            **kwargs
        )

    def emit_run(self, run_id: str, inputs: Dict, outputs: Dict,
                 status: str, metrics: Dict, upstream_urns: List[str] = None,
                 **kwargs) -> str:
        """Emit run metadata"""
        return self.run_emitter.emit_metadata(
            name=f"run_{run_id}",
            metadata={
                "id": run_id,
                "inputs": inputs,
                "outputs": outputs,
                "status": status,
                "metrics": metrics,
                **kwargs
            },
            description=f"LLM Run {run_id}",
            browse_paths=[
                "/llm/runs",
                f"/llm/runs/{status.lower()}",
                f"/llm/runs/id/{run_id}"
            ],
            entity_type="DATAJOB",
            sub_type="run",
            upstream_urns=upstream_urns,
            **kwargs
        )

    def _format_messages(self, messages) -> List[Dict]:
        """Format chat messages for metadata"""
        formatted = []
        for message in messages:
            message_type = message.__class__.__name__.lower().replace(
                "messageprompttemplate", ""
            )
            formatted.append({
                "role": message_type,
                "content": (
                    message.prompt.template
                    if hasattr(message.prompt, "template")
                    else str(message.prompt)
                ),
            })
        return formatted
