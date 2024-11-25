from typing import Dict, List, Optional, Any
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

class LangChainConnector(LLMPlatformConnector):
    """Connector for LangChain framework"""

    def __init__(self, group_models: bool = False):
        self.observed_models = {}
        self.observed_chains = {}
        self.observed_prompts = {}
        self.group_models = group_models  # Add flag to control model grouping

    def get_models(self) -> List[LLMModel]:
        """Get observed LangChain models"""
        return list(self.observed_models.values())

    def get_runs(self, **filters) -> List[LLMRun]:
        """Get run history - Note: LangChain doesn't store runs natively"""
        return []

    def get_chains(self) -> List[LLMChain]:
        """Get observed LangChain chains"""
        return list(self.observed_chains.values())

    def register_model(self, model: Any) -> None:
        """Register a LangChain model"""
        model_id = str(uuid.uuid4())
        self.observed_models[model_id] = self._create_model_from_langchain(model)

    def register_chain(self, chain: Any) -> None:
        """Register a LangChain chain"""
        chain_id = str(uuid.uuid4())
        self.observed_chains[chain_id] = self._create_chain_from_langchain(chain)

    def _create_model_from_langchain(self, model: Any) -> LLMModel:
        """Create LLMModel from LangChain model"""
        # Extract model info from the serialized data if it's a dict
        if isinstance(model, dict):
            model_name = model.get('model_name') or model.get('id', [])
        else:
            model_name = getattr(model, 'model_name', 'unknown')

        # Normalize the model name
        normalized_name = normalize_model_name(model_name)
        provider = get_provider_from_model(normalized_name)
        family = get_model_family(normalized_name)

        # Format display name based on grouping preference
        if self.group_models:
            display_name = f"{provider} {family}"  # e.g., "OpenAI GPT-3.5"
        else:
            display_name = f"{provider} {normalized_name}"  # e.g., "OpenAI gpt-3.5-turbo-0125"

        return LLMModel(
            name=display_name,
            provider=provider,
            model_family=family,
            capabilities=get_capabilities_from_model(normalized_name, model),
            parameters=get_model_parameters(model),
            metadata={
                "source": "langchain",
                "type": "chat" if "chat" in get_capabilities_from_model(normalized_name) else "completion",
                "description": f"{provider} {normalized_name} Language Model",
                "raw_name": normalized_name  # Store original name for reference
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
    """Observer for LangChain operations"""

    def __init__(self, config: ObservabilityConfig, emitter: LLMMetadataEmitter, group_models: bool = False):
        """Initialize observer with config and emitter"""
        self.config = config
        self.emitter = emitter
        self.active_runs: Dict[str, Dict] = {}
        self.connector = LangChainConnector(group_models=group_models)
        self.registered_models = set()  # Track registered models

    def _create_model_urn(self, model_identifier: str) -> str:
        """Create a DataHub URN for a model without using make_ml_model_urn"""
        # Format: urn:li:mlModel:(platform,name,env)
        platform = "openai"
        env = "PROD"
        return f"urn:li:mlModel:({platform},{model_identifier},{env})"

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running"""
        run_id = kwargs.get("run_id", str(uuid.uuid4()))

        if self.config.langchain_verbose:
            print(f"\nStarting LLM run: {run_id}")

        # Create and register model
        model = self.connector._create_model_from_langchain(serialized)
        model_id = f"{model.provider}/{model.name}"

        # Only emit model if we haven't seen it before
        if model_id not in self.registered_models:
            if self.config.langchain_verbose:
                print(f"Registering new model: {model_id}")
            self.log_model(model)
            self.registered_models.add(model_id)

        # Store model with run data
        self.active_runs[run_id] = {
            "start_time": datetime.now(),
            "prompts": prompts,
            "serialized": serialized,
            "model": model  # Store the model object
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
                model=run_data["model"],  # Use stored model object
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
