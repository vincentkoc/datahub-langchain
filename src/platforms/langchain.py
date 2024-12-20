from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
import requests

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, AgentAction
from langchain.prompts import BasePromptTemplate
from langchain_openai import ChatOpenAI

from ..base import (
    LLMObserver,
    LLMModel,
    LLMRun,
    LLMChain,
    LLMPlatformConnector,
    LLMMetadataEmitter
)
from ..models import Prompt, Metrics, Tool
from ..config import ObservabilityConfig
from ..utils.model_utils import get_capabilities_from_model, get_provider_from_model, get_model_family, get_model_parameters, normalize_model_name
from ..utils.pipeline_utils import detect_pipeline_name

# Add DataHub specific imports
from datahub.metadata.schema_classes import (
    MLModelSnapshotClass,
    MLModelPropertiesClass,
    GlobalTagsClass,
    TagAssociationClass,
    MLModelKeyClass,
    MLHyperParamClass,
    UpstreamLineageClass,
    UpstreamClass,
    MetadataChangeEventClass
)

class LangChainConnector(LLMPlatformConnector):
    """Connector for LangChain framework"""

    def __init__(self, group_models: bool = True):  # Default to True for hierarchical structure
        self.platform = "langchain"  # Change from 'llm' to 'langchain'
        self.observed_models = {}
        self.observed_runs = {}
        self.observed_chains = {}
        self.group_models = group_models

    def create_model_hierarchy(self, model: Any) -> Tuple[str, str]:
        """Create model with proper versioning and relationships"""
        model_info = self._create_model_from_langchain(model)

        # Create model URN with specific version
        model_urn = make_ml_model_urn(
            platform="langchain",
            model_name=model_info.name,  # e.g. "OpenAI gpt-3.5-turbo-0125"
            env="PROD"
        )

        # Create model properties with full metadata
        properties = MLModelPropertiesClass(
            description=model_info.metadata["description"],
            type="Language Model",
            customProperties={
                "provider": model_info.provider,
                "model_family": model_info.model_family,
                "capabilities": str(model_info.capabilities),
                "parameters": str(model_info.parameters),
                "platform": "langchain",
                **model_info.metadata
            },
            hyperParams=[
                MLHyperParamClass(name=k, value=str(v))
                for k, v in model_info.parameters.items()
            ]
        )

        # Add metrics as custom properties instead of using MLModelMetricsClass
        if hasattr(model_info, "metrics"):
            properties.customProperties.update({
                "training_metrics": str(model_info.metrics.get("training", [])),
                "evaluation_metrics": str(model_info.metrics.get("evaluation", [])),
                "hyperparameters": str(model_info.parameters)
            })

        return model_urn

    def register_model(self, model: Any) -> None:
        """Register a LangChain model with hierarchical structure"""
        group, instance = self.create_model_hierarchy(model)

        # Store both group and instance
        group_id = group.urn.split(":")[-1]
        if group_id not in self.observed_model_groups:
            self.observed_model_groups[group_id] = group

        instance_id = instance.urn.split(":")[-1]
        self.observed_model_runs[instance_id] = instance

    def get_models(self) -> List[LLMModel]:
        """Get observed LangChain models"""
        # Return both groups and instances
        models = []
        for group in self.observed_model_groups.values():
            models.append(self._convert_group_to_model(group))
        for instance in self.observed_model_runs.values():
            models.append(self._convert_instance_to_model(instance))
        return models

    def _convert_group_to_model(self, group: MLModelSnapshotClass) -> LLMModel:
        """Convert MLModelSnapshot to LLMModel"""
        # Get properties from the first aspect (MLModelPropertiesClass)
        props = next((aspect for aspect in group.aspects if isinstance(aspect, MLModelPropertiesClass)), None)
        if not props:
            raise ValueError("Missing MLModelProperties aspect")

        # Get tags from the second aspect (GlobalTagsClass)
        tags = next((aspect for aspect in group.aspects if isinstance(aspect, GlobalTagsClass)), None)

        return LLMModel(
            name=group.urn.split(":")[-1],
            provider=props.customProperties.get("provider", "unknown"),
            model_family=props.customProperties.get("model_family", "unknown"),
            capabilities=eval(props.customProperties.get("capabilities", "[]")),
            parameters={},
            metadata={"type": "model_group"}
        )

    def _convert_instance_to_model(self, instance: MLModelSnapshotClass) -> LLMModel:
        """Convert MLModelSnapshot to LLMModel"""
        # Get properties from the first aspect (MLModelPropertiesClass)
        props = next((aspect for aspect in instance.aspects if isinstance(aspect, MLModelPropertiesClass)), None)
        if not props:
            raise ValueError("Missing MLModelProperties aspect")

        # Get tags from the second aspect (GlobalTagsClass)
        tags = next((aspect for aspect in instance.aspects if isinstance(aspect, GlobalTagsClass)), None)

        return LLMModel(
            name=instance.urn.split(":")[-1],
            provider=props.customProperties.get("provider", "unknown"),
            model_family=props.customProperties.get("model_family", "unknown"),
            capabilities=[tag.tag.split(":")[-1] for tag in tags.tags] if tags else [],
            parameters={k: v for k, v in props.customProperties.items()
                       if k not in ["provider", "model_family", "group_urn"]},
            metadata={"type": "model_instance"}
        )

    def get_runs(self, **filters) -> List[LLMRun]:
        """Get run history - Note: LangChain doesn't store runs natively"""
        return []

    def get_chains(self) -> List[LLMChain]:
        """Get observed LangChain chains"""
        return list(self.observed_chains.values())

    def _create_model_from_langchain(self, model_info: Any) -> LLMModel:
        """Create LLMModel from LangChain model"""
        try:
            # Extract model name with better fallbacks
            if isinstance(model_info, dict):
                model_name = (
                    model_info.get("model_name") or
                    model_info.get("model") or
                    model_info.get("name", "unknown")
                )
            else:
                # For ChatOpenAI and similar objects
                model_name = (
                    getattr(model_info, "model_name", None) or  # This should get "gpt-3.5-turbo"
                    getattr(model_info, "model", None) or
                    getattr(model_info, "name", None) or
                    model_info.__class__.__name__.lower()  # Last resort: use class name
                )

            # Get raw name for reference (class name)
            raw_name = (
                model_info.__class__.__name__
                if not isinstance(model_info, dict)
                else model_info.get("class_name", "Unknown")
            )

            # Clean up model name and get metadata
            model_name = normalize_model_name(model_name)  # This will convert ChatOpenAI to gpt-3.5-turbo
            provider = get_provider_from_model(model_name)  # Will return "OpenAI"
            model_family = get_model_family(model_name)  # Will return "GPT-3.5"
            capabilities = get_capabilities_from_model(model_name)
            parameters = get_model_parameters(model_info)

            # Create descriptive name and description
            display_name = f"{provider} {model_name}"  # e.g. "OpenAI gpt-3.5-turbo"
            description = f"{provider} {model_family} Model ({model_name})"  # e.g. "OpenAI GPT-3.5 Model (gpt-3.5-turbo)"

            return LLMModel(
                name=model_name,  # Use actual model name (gpt-3.5-turbo)
                provider=provider,
                model_family=model_family,
                capabilities=capabilities,
                parameters=parameters,
                metadata={
                    "raw_name": raw_name,  # Store class name here (ChatOpenAI)
                    "source": "langchain",
                    "display_name": display_name,
                    "description": description
                }
            )
        except Exception as e:
            print(f"Error creating model from LangChain: {e}")
            return LLMModel(
                name="unknown_model",
                provider="Unknown",
                model_family="Unknown",
                capabilities=["text-generation"],
                parameters={},
                metadata={
                    "error": str(e),
                    "source": "langchain",
                    "description": "Unknown LangChain Model (Error during creation)"
                }
            )

    def _create_chain_from_langchain(self, chain: Any) -> LLMChain:
        """Create LLMChain from LangChain chain"""
        return LLMChain(
            id=str(uuid.uuid4()),
            name=chain.__class__.__name__,
            components=self._get_chain_components(chain),
            config=self._get_chain_config(chain),
            metadata={"source": "langchain"}
        )

    @staticmethod
    def _get_family_from_model(model_name: str) -> str:
        """Determine model family from name"""
        if 'gpt-4' in model_name:
            return 'GPT-4'
        elif 'gpt-3.5' in model_name:
            return 'GPT-3.5'
        elif 'claude' in model_name:
            return 'Claude'
        elif 'llama' in model_name:
            return 'LLaMA'
        return 'GPT'  # Default family for OpenAI models

    @staticmethod
    def _get_model_parameters(model: Any) -> Dict[str, Any]:
        """Get model parameters"""
        params = {
            "contextWindow": 4096,  # Default for GPT-3.5
            "tokenLimit": 4096,
            "costPerToken": 0.0001
        }

        if isinstance(model, dict):
            for param in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
                if param in model:
                    params[param] = model[param]
        else:
            for param in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
                if hasattr(model, param):
                    params[param] = getattr(model, param)
        return params

    @staticmethod
    def _get_chain_components(chain: Any) -> List[str]:
        """Get chain components"""
        components = []
        if hasattr(chain, 'llm'):
            components.append('llm')
        if hasattr(chain, 'prompt'):
            components.append('prompt')
        if hasattr(chain, 'memory'):
            components.append('memory')
        return components

    @staticmethod
    def _get_chain_config(chain: Any) -> Dict[str, Any]:
        """Get chain configuration"""
        return {
            "verbose": getattr(chain, 'verbose', False),
            "chain_type": chain.__class__.__name__,
            "has_memory": hasattr(chain, 'memory')
        }

class LangChainObserver(BaseCallbackHandler, LLMObserver):
    """LangChain observer that emits metadata to DataHub"""

    def __init__(self, config: ObservabilityConfig, emitter: LLMMetadataEmitter,
                 pipeline_name: Optional[str] = None, group_models: bool = True,
                 hard_fail: bool = False):
        # Call both parent class initializers
        BaseCallbackHandler.__init__(self)
        LLMObserver.__init__(self)

        self.config = config
        self.emitter = emitter
        self.active_runs: Dict[str, Dict] = {}
        self.connector = LangChainConnector(group_models=group_models)
        self.hard_fail = hard_fail

        # Use provided pipeline name or get from source file
        self.pipeline_name = pipeline_name if pipeline_name else detect_pipeline_name()

        # Store pipeline name in active runs for emission
        self._pipeline_metadata = {
            "pipeline_name": self.pipeline_name,
            "framework": "langchain",
            "source": "langchain"
        }

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs) -> None:
        run_id = kwargs.get("run_id", str(uuid.uuid4()))

        if self.config.langchain_verbose:
            print(f"\nStarting LLM run in pipeline {self.pipeline_name}: {run_id}")

        # Store pipeline info with run data
        self.active_runs[run_id] = {
            "start_time": datetime.now(),
            "prompts": prompts,
            "serialized": serialized,
            "pipeline_name": self.pipeline_name,
            "model_info": self.connector._create_model_from_langchain(serialized)
        }

        self.start_run(run_id)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends running"""
        run_id = kwargs.get("run_id")
        if run_id in self.active_runs:
            run_data = self.active_runs[run_id]

            if self.config.langchain_verbose:
                print(f"\nProcessing LLM run end: {run_id}")

            metrics = Metrics(
                latency=(datetime.now() - run_data["start_time"]).total_seconds(),
                token_usage=response.llm_output.get("token_usage", {}) if response.llm_output else {},
                cost=response.llm_output.get("cost", 0) if response.llm_output else 0,
                error_rate=0.0,
                success_rate=1.0,
                custom_metrics={}
            )

            run = LLMRun(
                id=run_id,
                start_time=run_data["start_time"],
                end_time=datetime.now(),
                model=run_data["model_info"],
                inputs={"prompts": run_data["prompts"]},
                outputs={"generations": [g.text for g in response.generations[0]]},
                metrics=metrics.__dict__,
                parent_id=kwargs.get("parent_run_id"),
                metadata={
                    **run_data["serialized"],
                    **self._pipeline_metadata,  # Include pipeline metadata
                    "handler": self.config.langchain_handler
                }
            )

            if self.config.langchain_verbose:
                print(f"Completed LLM run: {run_id}")
                print(f"Latency: {metrics.latency:.2f}s")
                if metrics.token_usage:
                    print(f"Tokens: {metrics.token_usage}")
                print(f"\nEmitting run to DataHub...")

            # First emit the model
            if run.model:
                model_urn = self.emitter.emit_model(run.model)
                if self.config.langchain_verbose:
                    print(f"Emitted model: {model_urn}")

            # Then emit the run with lineage
            pipeline_urn = self.emit_run(run)
            if self.config.langchain_verbose:
                print(f"Emitted pipeline: {pipeline_urn}")

            self.end_run(run_id)
            del self.active_runs[run_id]

    def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs) -> None:
        """Called when chain starts running"""
        chain = self.connector._create_chain_from_langchain(serialized)
        self.log_chain(chain)

    def start_run(self, run_id: str, **kwargs) -> None:
        """Start observing a run"""
        if self.config.langchain_verbose:
            print(f"Starting run: {run_id}")

    def end_run(self, run_id: str, **kwargs) -> None:
        """End observing a run"""
        if self.config.langchain_verbose:
            print(f"Ending run: {run_id}")

    def log_metrics(self, run_id: str, metrics: Dict[str, Any]) -> None:
        """Log metrics for a run"""
        if run_id in self.active_runs:
            self.active_runs[run_id]["metrics"] = metrics

    def log_model(self, model: LLMModel) -> None:
        """Log model metadata"""
        self.emitter.emit_model(model)

    def log_chain(self, chain: LLMChain) -> None:
        """Log chain metadata"""
        self.emitter.emit_chain(chain)

    def create_llm_hierarchy(self, model_name: str, run_id: str):
        # Create model group
        group = MLModelSnapshotClass(
            urn=f"urn:li:mlModelGroup:(urn:li:dataPlatform:llm,{model_name},PROD)",
            aspects=[
                MLModelPropertiesClass(
                    description=f"LLM Group for {model_name}"
                )
            ]
        )

        # Create run as model instance
        run_model = MLModelSnapshotClass(
            urn=f"urn:li:mlModel:(urn:li:dataPlatform:llm,{model_name}_run_{run_id},PROD)",
            aspects=[
                MLModelPropertiesClass(
                    description=f"Run instance of {model_name}",
                    groups=[group.urn]
                )
            ]
        )

        return group, run_model

    def emit_run(self, run: LLMRun) -> str:
        """Emit run metadata as part of a pipeline and create lineage"""
        try:
            # First emit model and get URN
            model_urn = None
            if run.model:
                model_urn = self.emitter.emit_model(run.model)
                if self.config.langchain_verbose:
                    print(f"Emitted model with URN: {model_urn}")

            # Create pipeline metadata
            pipeline_name = run.metadata.get("pipeline_name") or detect_pipeline_name()
            pipeline_urn = f"urn:li:mlModel:(urn:li:dataPlatform:langchain,{pipeline_name},PROD)"

            # Create properties with lineage through trainingJobs
            properties = MLModelPropertiesClass(
                description=f"LangChain Pipeline: {pipeline_name}",
                type="Pipeline",
                customProperties={
                    "pipeline_name": pipeline_name,
                    "framework": "langchain",
                    "last_run_id": str(run.id),
                    "last_run_time": str(run.start_time.isoformat()),
                    "last_run_status": "completed" if not run.metrics.get("error") else "failed",
                    "latency": str(run.metrics.get("latency", 0)),
                    "total_tokens": str(run.metrics.get("token_usage", {}).get("total_tokens", 0)),
                    "cost": str(run.metrics.get("cost", 0)),
                    "error_rate": str(run.metrics.get("error_rate", 0)),
                    "success_rate": str(run.metrics.get("success_rate", 1.0))
                },
                trainingJobs=[model_urn] if model_urn else None  # Add lineage through trainingJobs
            )

            # Add tags
            tags = GlobalTagsClass(
                tags=[
                    TagAssociationClass(tag=f"urn:li:tag:pipeline"),
                    TagAssociationClass(tag=f"urn:li:tag:langchain"),
                    TagAssociationClass(tag=f"urn:li:tag:{'success' if not run.metrics.get('error') else 'failed'}")
                ]
            )

            # Create and emit pipeline MCE with lineage
            pipeline_mce = MetadataChangeEventClass(
                proposedSnapshot=MLModelSnapshotClass(
                    urn=pipeline_urn,
                    aspects=[properties, tags]
                )
            )

            if self.config.langchain_verbose:
                print(f"Emitting pipeline MCE with {len([properties, tags])} aspects")

            # Emit pipeline with lineage
            self.emitter._emit_with_retry(pipeline_mce)

            return pipeline_urn

        except Exception as e:
            if self.config.langchain_verbose:
                print(f"\n✗ Failed to emit pipeline run {run.id}: {e}")
            if self.hard_fail:
                raise
            return ""
