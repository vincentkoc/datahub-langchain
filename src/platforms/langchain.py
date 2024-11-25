from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid

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

# Add DataHub specific imports
from datahub.metadata.schema_classes import (
    MLModelSnapshotClass,
    MLModelPropertiesClass,
    GlobalTagsClass,
    TagAssociationClass,
    MLModelKeyClass,
    MLHyperParamClass
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

    def _create_model_from_langchain(self, model: Any) -> LLMModel:
        """Create LLMModel from LangChain model with proper versioning"""
        if isinstance(model, dict):
            model_name = model.get('model_name') or model.get('id', [])
        else:
            model_name = getattr(model, 'model_name', 'unknown')

        normalized_name = normalize_model_name(model_name)
        provider = get_provider_from_model(normalized_name)
        family = get_model_family(normalized_name)

        # Use specific version name instead of generic family
        display_name = f"{provider} {normalized_name}"  # e.g. "OpenAI gpt-3.5-turbo-0125"

        return LLMModel(
            name=display_name,
            provider=provider,
            model_family=family,
            capabilities=get_capabilities_from_model(normalized_name, model),
            parameters=get_model_parameters(model),
            metadata={
                "source": "langchain",
                "platform": "langchain",
                "type": "chat" if "chat" in get_capabilities_from_model(normalized_name) else "completion",
                "description": f"{provider} {normalized_name} Language Model",
                "raw_name": normalized_name
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

    def register_platform(self):
        """Register LangChain as a DataHub platform using available classes"""
        # Create platform metadata using MLModelProperties instead
        platform_props = MLModelPropertiesClass(
            description="LangChain Framework for LLM Applications",
            type="FRAMEWORK",
            customProperties={
                "name": "LangChain",
                "displayName": "LangChain",
                "type": "llm_framework",
                "domain": "ai_ml",
                "category": "development_framework",
                "logoUrl": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/logos/langchain-ipuhh4qo1jz5ssl4x0g2a.png/langchain-dp1uxj2zn3752pntqnpfu2.png?_a=DAJFJtWIZAAC"
            }
        )

        # Create platform snapshot
        mce = MetadataChangeEventClass(
            proposedSnapshot=MLModelSnapshotClass(
                urn="urn:li:mlModel:(urn:li:dataPlatform:langchain,platform,PROD)",
                aspects=[platform_props]
            )
        )

        self.emitter.emit(mce)

class LangChainObserver(BaseCallbackHandler, LLMObserver):
    """Observer for LangChain operations"""

    def __init__(self, config: ObservabilityConfig, emitter: LLMMetadataEmitter,
                 pipeline_name: Optional[str] = None, group_models: bool = False):
        """Initialize observer with config and emitter"""
        self.config = config
        self.emitter = emitter
        self.active_runs: Dict[str, Dict] = {}
        self.connector = LangChainConnector(group_models=group_models)
        self.registered_models = set()

        # Get pipeline name from source file or override
        self.pipeline_name = pipeline_name or self._detect_pipeline_name()

    def _detect_pipeline_name(self) -> str:
        """Detect pipeline name from calling file"""
        import inspect
        import os

        # Get the calling frame (skip current frame and LangChain internals)
        frame = inspect.currentframe()
        while frame:
            filename = frame.f_code.co_filename
            if not any(x in filename for x in ['langchain', 'datahub_langchain']):
                break
            frame = frame.f_back

        if frame:
            # Get filename without extension and path
            pipeline = os.path.splitext(os.path.basename(filename))[0]
            return pipeline
        return "default_pipeline"

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
                model=run_data["model_info"],  # Use stored model object
                inputs={"prompts": run_data["prompts"]},
                outputs={"generations": [g.text for g in response.generations[0]]},
                metrics=metrics.__dict__,
                parent_id=kwargs.get("parent_run_id"),
                metadata={
                    **run_data["serialized"],
                    "handler": self.config.langchain_handler
                }
            )

            if self.config.langchain_verbose:
                print(f"Completed LLM run: {run_id}")
                print(f"Latency: {metrics.latency:.2f}s")
                if metrics.token_usage:
                    print(f"Tokens: {metrics.token_usage}")

            self.emitter.emit_run(run)
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
