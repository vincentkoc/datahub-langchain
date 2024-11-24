from typing import Dict, List, Optional
from datetime import datetime
import json

from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    MLModelPropertiesClass,
    StatusClass,
    BrowsePathsClass,
    DatasetSnapshotClass,
    MLModelSnapshotClass,
    MetadataChangeEventClass,
)

from .metadata_emitter import BaseMetadataEmitter

class LLMMetadataEmitter:
    """Metadata emitter for LangChain/LangSmith components"""

    def __init__(self, gms_server: str = None):
        # Use langchain as the main platform
        self.model_emitter = BaseMetadataEmitter("langchain", gms_server)
        self.prompt_emitter = BaseMetadataEmitter("langchain", gms_server)
        self.pipeline_emitter = BaseMetadataEmitter("langchain", gms_server)
        self.run_emitter = BaseMetadataEmitter("langsmith", gms_server)  # LangSmith specific
        self._emitted_urns = set()  # Track emitted URNs
        self.source_info = {
            "ingestion_source": "langchain_datahub",
            "ingestion_time": datetime.now().isoformat(),
            "version": "1.0"
        }

    @property
    def is_dry_run(self) -> bool:
        return self.model_emitter.is_dry_run

    def _check_duplicate_urn(self, urn: str, metadata: Dict) -> bool:
        """Check if URN already exists and compare metadata"""
        if urn in self._emitted_urns:
            print(f"Warning: Duplicate URN detected: {urn}")
            # TODO: Could add metadata comparison here
            return True
        self._emitted_urns.add(urn)
        return False

    def _add_source_tracking(self, metadata: Dict, source: str = None) -> Dict:
        """Add source tracking to metadata"""
        source_metadata = {
            **self.source_info,
            "specific_source": source or "unknown",
            "discovery_path": "direct" if source == "langchain" else "discovered"
        }

        if "metadata" not in metadata:
            metadata["metadata"] = {}

        metadata["metadata"]["source_tracking"] = source_metadata
        return metadata

    def emit_metadata(self, name: str, metadata: Dict, source: str = None, **kwargs) -> Optional[str]:
        """Emit metadata with duplicate and source tracking"""
        # Add source tracking
        metadata = self._add_source_tracking(metadata, source)

        # Generate URN (you'll need to implement this based on your URN structure)
        urn = self._generate_urn(name, metadata)

        # Check for duplicates
        if self._check_duplicate_urn(urn, metadata):
            print(f"Skipping duplicate emission for {urn}")
            return None

        # Add to metadata
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["emission_count"] = len(self._emitted_urns)

        return super().emit_metadata(name=name, metadata=metadata, **kwargs)

    def emit_model(self, model_name: str, provider: str, model_type: str,
                  capabilities: List[str], parameters: Dict, **kwargs) -> str:
        """Emit LLM model metadata"""
        return self.model_emitter.emit_metadata(
            name=f"{provider.lower()}/{model_name}",  # e.g. openai/gpt-4
            metadata={
                "provider": provider,
                "model_type": model_type,
                "capabilities": capabilities,
                "parameters": parameters
            },
            description=f"{provider} {model_name} Language Model",
            browse_paths=[
                "/langchain/models",
                f"/langchain/models/{provider.lower()}",
                f"/langchain/models/{provider.lower()}/{model_name}"
            ],
            entity_type="MLMODEL",
            sub_type="llm_model",
            icon_url="https://raw.githubusercontent.com/datahub-project/datahub/master/datahub-web-react/src/images/openai.png",
            **kwargs
        )

    def emit_prompt(self, prompt, template_format: str, category: str,
                   version: str, **kwargs) -> str:
        """Emit prompt metadata"""
        formatted_messages = self._format_messages(prompt.messages)
        prompt_id = hash(str(formatted_messages))

        return self.prompt_emitter.emit_metadata(
            name=f"prompts/{category.lower()}/{prompt_id}",
            metadata={
                "template": formatted_messages,
                "format": template_format,
                "category": category,
                "version": version,
                "input_variables": list(prompt.input_variables)
            },
            description="LangChain Prompt Template",
            browse_paths=[
                "/langchain/prompts",
                f"/langchain/prompts/{category.lower()}",
                f"/langchain/prompts/{category.lower()}/{prompt_id}"
            ],
            entity_type="DATASET",
            sub_type="prompt_template",
            icon_url="https://raw.githubusercontent.com/datahub-project/datahub/master/datahub-web-react/src/images/langchain.png",
            **kwargs
        )

    def emit_pipeline(self, chain_type: str, config: Dict, upstream_urns: List[str],
                     source: str = "langchain", **kwargs) -> str:
        """Emit pipeline metadata with source tracking"""
        metadata = {
            "type": chain_type,
            "config": config,
            "source_info": {
                "discovered_by": source,
                "discovery_time": datetime.now().isoformat(),
                "chain_source": source
            }
        }

        return self.pipeline_emitter.emit_metadata(
            name=f"pipelines/{chain_type.lower()}",
            metadata=metadata,
            description=f"LangChain Pipeline: {chain_type}",
            browse_paths=[
                "/langchain/pipelines",
                f"/langchain/pipelines/{chain_type.lower()}"
            ],
            entity_type="DATASET",  # Keep as DATASET since DATAPROCESS isn't fully supported
            sub_type="pipeline",
            upstream_urns=upstream_urns,
            icon_url="https://raw.githubusercontent.com/datahub-project/datahub/master/datahub-web-react/src/images/langchain.png",
            **kwargs
        )

    def emit_run(self, run_id: str, inputs: Dict, outputs: Dict, status: str, metrics: Dict,
                 upstream_urns: List[str] = None, error: str = None, tags: List[str] = None,
                 feedback_stats: Dict = None, source: str = "langsmith", **kwargs) -> str:
        """Emit run metadata with source tracking"""
        metadata = {
            # Core properties
            "name": f"Run {run_id}",
            "description": f"LangSmith Run {run_id}",
            "runId": run_id,
            "status": status,

            # Input/Output properties
            "inputProperties": {
                "type": "object",
                "properties": inputs
            },
            "outputProperties": {
                "type": "object",
                "properties": outputs
            },

            # Performance metrics as separate properties
            "tokenUsage": metrics.get("tokenUsage", {}),
            "latency": metrics.get("latency", 0),
            "cost": metrics.get("cost", 0),

            # Error tracking
            "errorMessage": error if error else None,
            "errorType": error.__class__.__name__ if error else None,

            # Feedback and evaluation
            "feedbackStats": feedback_stats if feedback_stats else {},

            # Lineage tracking
            "upstreamUrns": upstream_urns if upstream_urns else [],

            # Tags and metadata
            "tags": tags if tags else [],
            "created": datetime.now().isoformat(),

            # Custom properties (moved from separate argument)
            "customProperties": {
                key: json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                for key, value in {
                    "tokenUsage": metrics.get("tokenUsage", {}),
                    "latency": metrics.get("latency", 0),
                    "cost": metrics.get("cost", 0),
                    "error": error if error else None,
                    "tags": tags if tags else [],
                    "feedback": feedback_stats if feedback_stats else {}
                }.items()
            },
            "source_info": {
                "discovered_by": source,
                "discovery_time": datetime.now().isoformat(),
                "run_source": "langsmith" if source == "langsmith" else "langchain"
            }
        }

        return self.run_emitter.emit_metadata(
            name=f"runs/{run_id}",
            metadata=metadata,
            description=f"LangSmith Run {run_id}",
            browse_paths=[
                "/langsmith/runs",
                f"/langsmith/runs/status/{status.lower()}",
                f"/langsmith/runs/id/{run_id}"
            ],
            entity_type="DATASET",  # Keep as DATASET since DATAPROCESS isn't fully supported
            sub_type="run",
            upstream_urns=upstream_urns,
            tags=tags,
            icon_url="https://raw.githubusercontent.com/datahub-project/datahub/master/datahub-web-react/src/images/langchain.png",
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

    def emit_lineage(self, source_urn: str, target_urn: str, lineage_type: str = "Produces"):
        """Emit explicit lineage between entities"""
        try:
            # Determine entity types
            source_is_dataset = "dataset" in source_urn
            target_is_dataset = "dataset" in target_urn

            # Emit upstream lineage
            upstream_mce = MetadataChangeEventClass(
                proposedSnapshot=(DatasetSnapshotClass if source_is_dataset else MLModelSnapshotClass)(
                    urn=source_urn,
                    aspects=[{
                        "com.linkedin.metadata.aspect.UpstreamLineage": {
                            "upstreams": [{
                                "auditStamp": {
                                    "time": int(datetime.now().timestamp() * 1000),
                                    "actor": "urn:li:corpuser:datahub"
                                },
                                "dataset": {
                                    "entityType": "dataset" if target_is_dataset else "mlModel",
                                    "urn": target_urn
                                },
                                "type": lineage_type
                            }]
                        }
                    }]
                )
            )

            # Emit downstream lineage
            downstream_mce = MetadataChangeEventClass(
                proposedSnapshot=(DatasetSnapshotClass if target_is_dataset else MLModelSnapshotClass)(
                    urn=target_urn,
                    aspects=[{
                        "com.linkedin.metadata.aspect.DownstreamLineage": {
                            "downstreams": [{
                                "auditStamp": {
                                    "time": int(datetime.now().timestamp() * 1000),
                                    "actor": "urn:li:corpuser:datahub"
                                },
                                "dataset": {
                                    "entityType": "dataset" if source_is_dataset else "mlModel",
                                    "urn": source_urn
                                },
                                "type": lineage_type
                            }]
                        }
                    }]
                )
            )

            print(f"\nEmitting lineage:")
            print(f"Source: {source_urn}")
            print(f"Target: {target_urn}")
            print(f"Type: {lineage_type}")

            # Emit both aspects
            self.run_emitter.emitter.emit(upstream_mce)
            self.run_emitter.emitter.emit(downstream_mce)
            print("Successfully emitted lineage")

        except Exception as e:
            print(f"Failed to emit lineage: {e}")
            if not self.run_emitter.is_dry_run:
                raise

    def emit_complete_run(self, run, model, chain):
        """Emit complete run with all relationships"""

        # Emit model metadata
        model_urn = self.emit_model(
            model_name=model.model_name,
            provider="OpenAI",
            capabilities=["text-generation"],
            parameters={
                "temperature": model.temperature,
                "maxTokens": model.max_tokens
            }
        )

        # Emit chain metadata
        chain_urn = self.emit_pipeline(
            chain_type=chain.__class__.__name__,
            config={
                "type": "sequential",
                "components": ["prompt", "model"]
            },
            upstream_urns=[model_urn]
        )

        # Emit run metadata
        run_urn = self.emit_run(
            run_id=run.id,
            inputs=run.inputs,
            outputs=run.outputs,
            status=run.status,
            metrics={
                "tokenUsage": run.token_usage,
                "latency": run.latency,
                "cost": run.cost
            },
            upstream_urns=[chain_urn]
        )

        # Emit explicit lineage
        self.emit_lineage(run_urn, chain_urn, "ExecutedBy")
        self.emit_lineage(chain_urn, model_urn, "Uses")

        return run_urn

    def add_search_aspects(self, urn: str, search_fields: Dict):
        """Add searchable aspects to entity"""
        mce = MetadataChangeEventClass(
            proposedSnapshot=DatasetSnapshotClass(
                urn=urn,
                aspects=[{
                    "com.linkedin.metadata.aspect.SearchableFieldsAspect": {
                        "fields": [
                            {
                                "fieldName": key,
                                "fieldValue": value,
                                "fieldType": "KEYWORD"
                            }
                            for key, value in search_fields.items()
                        ]
                    }
                }]
            )
        )
        self.emitter.emit(mce)

class LLMRunAspect:
    def __init__(self, run_id: str, status: str, metrics: Dict):
        self.run_id = run_id
        self.status = status
        self.metrics = metrics

    def to_obj(self) -> Dict:
        return {
            "com.linkedin.metadata.aspect.LLMRunAspect": {
                "runId": self.run_id,
                "status": self.status,
                "metrics": {
                    "tokenUsage": self.metrics.get("tokenUsage", {}),
                    "latency": self.metrics.get("latency", 0),
                    "cost": self.metrics.get("cost", 0)
                }
            }
        }

class LLMModelAspect:
    def __init__(self, model_name: str, provider: str, capabilities: List[str]):
        self.model_name = model_name
        self.provider = provider
        self.capabilities = capabilities

    def to_obj(self) -> Dict:
        return {
            "com.linkedin.metadata.aspect.LLMModelAspect": {
                "modelName": self.model_name,
                "provider": self.provider,
                "capabilities": self.capabilities
            }
        }
