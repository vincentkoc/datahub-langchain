from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID
import json
from pathlib import Path

from langsmith import Client
from .base import BaseIngestor
from ..base import (
    LLMPlatformConnector,
    LLMModel,
    LLMRun,
    LLMChain,
)
from ..models import Metrics
from ..config import ObservabilityConfig
from src.utils.model_utils import get_capabilities_from_model

class LangSmithConnector(LLMPlatformConnector):
    """Connector for LangSmith platform"""

    def __init__(self, config: ObservabilityConfig):
        """Initialize LangSmith connector with config"""
        self.config = config
        langsmith_config = config.get_platform_config("langsmith")

        self.client = Client(
            api_url=langsmith_config["endpoint"],
            api_key=langsmith_config["api_key"]
        )
        self.project_name = langsmith_config["project"]

    def get_models(self) -> List[LLMModel]:
        """Get models from LangSmith runs"""
        try:
            # Get unique models from recent runs
            runs = self.client.list_runs(
                project_name=self.project_name,
                execution_order=1,
                start_time=datetime.now() - timedelta(days=self.config.ingest_window_days)
            )

            models = {}
            for run in runs:
                if hasattr(run, 'execution_metadata') and run.execution_metadata:
                    model_name = run.execution_metadata.get('model_name')
                    if model_name and model_name not in models:
                        models[model_name] = self._create_model_from_run(run)

            return list(models.values())
        except Exception as e:
            print(f"Error fetching LangSmith models: {e}")
            return []

    def get_runs(self, start_time: datetime = None, end_time: datetime = None,
                limit: int = 1000) -> List[LLMRun]:
        """Get run history from LangSmith"""
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(days=7)

            runs = self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                end_time=end_time,
                execution_order=1,
                limit=limit
            )

            return [self._create_run_from_langsmith(run) for run in runs]
        except Exception as e:
            print(f"Error fetching LangSmith runs: {e}")
            return []

    def get_chains(self) -> List[LLMChain]:
        """Get chain definitions from LangSmith"""
        try:
            # Get unique chains from recent runs
            runs = self.client.list_runs(
                project_name=self.project_name,
                execution_order=1,
                start_time=datetime.now() - timedelta(days=7)
            )

            chains = {}
            for run in runs:
                if getattr(run, 'run_type', '') == 'chain':
                    chain_id = getattr(run, 'id', str(UUID()))
                    if chain_id not in chains:
                        chains[chain_id] = self._create_chain_from_run(run)

            return list(chains.values())
        except Exception as e:
            print(f"Error fetching LangSmith chains: {e}")
            return []

    def _create_model_from_run(self, run) -> LLMModel:
        """Create LLMModel from LangSmith run"""
        execution_metadata = getattr(run, 'execution_metadata', {}) or {}
        model_name = execution_metadata.get('model_name', 'unknown')

        return LLMModel(
            name=model_name,
            provider=self._get_provider_from_model(model_name),
            model_family=self._get_family_from_model(model_name),
            capabilities=get_capabilities_from_model(model_name),
            parameters={
                "context_window": execution_metadata.get('context_window', 4096),
                "max_tokens": execution_metadata.get('max_tokens', 4096),
            },
            metadata={
                "source": "langsmith",
                "raw_metadata": execution_metadata
            }
        )

    def _create_run_from_langsmith(self, run) -> LLMRun:
        """Create LLMRun from LangSmith run"""
        metrics = Metrics(
            latency=getattr(run, 'latency', 0),
            token_usage=getattr(run, 'token_usage', {}),
            cost=getattr(run, 'cost', 0),
            error_rate=0.0,  # Would need to calculate from historical data
            success_rate=1.0 if getattr(run, 'error', None) is None else 0.0,
            custom_metrics={}
        )

        return LLMRun(
            id=str(run.id),
            start_time=getattr(run, 'start_time', datetime.now()),
            end_time=getattr(run, 'end_time', None),
            model=self._create_model_from_run(run) if hasattr(run, 'execution_metadata') else None,
            inputs=getattr(run, 'inputs', {}),
            outputs=getattr(run, 'outputs', {}),
            metrics=metrics.__dict__,
            parent_id=getattr(run, 'parent_run_id', None),
            metadata={
                "run_type": getattr(run, 'run_type', 'unknown'),
                "error": str(run.error) if hasattr(run, 'error') and run.error else None,
                "tags": list(getattr(run, 'tags', [])),
                "feedback_stats": getattr(run, 'feedback_stats', {})
            }
        )

    def _create_chain_from_run(self, run) -> LLMChain:
        """Create LLMChain from LangSmith run"""
        return LLMChain(
            id=str(run.id),
            name=getattr(run, 'name', 'unknown'),
            components=self._extract_chain_components(run),
            config={
                "max_retries": 3,
                "verbose": False,
                "execution_metadata": getattr(run, 'execution_metadata', {}),
                "run_type": getattr(run, 'run_type', 'unknown'),
                "serialized": getattr(run, 'serialized', {})
            },
            metadata={
                "source": "langsmith",
                "project": self.project_name,
                "discovery_time": datetime.now().isoformat()
            }
        )

    def _extract_chain_components(self, run) -> List[str]:
        """Extract chain components from run"""
        components = []
        if hasattr(run, 'child_runs'):
            for child in run.child_runs:
                components.append(getattr(child, 'run_type', 'unknown'))
        return components

    @staticmethod
    def _get_provider_from_model(model_name: str) -> str:
        """Determine provider from model name"""
        if any(x in model_name.lower() for x in ['gpt', 'davinci', 'curie']):
            return 'OpenAI'
        elif 'claude' in model_name.lower():
            return 'Anthropic'
        return 'unknown'

    @staticmethod
    def _get_family_from_model(model_name: str) -> str:
        """Determine model family from name"""
        if 'gpt-4' in model_name:
            return 'GPT-4'
        elif 'gpt-3.5' in model_name:
            return 'GPT-3.5'
        elif 'claude' in model_name:
            return 'Claude'
        return 'unknown'

class LangsmithIngestor(BaseIngestor):
    def __init__(self, config, save_debug_data=True, processing_dir=None):
        self.connector = LangSmithConnector(config)
        self.save_debug_data = save_debug_data
        self.processing_dir = processing_dir

    def fetch_data(self):
        raw_data = self.connector.get_runs()
        if self.save_debug_data and self.processing_dir:
            raw_data_path = self.processing_dir / 'langsmith_api_output.json'
            with open(raw_data_path, 'w') as f:
                # Assuming that LLMRun has a method to_dict() or similar
                json.dump([run.__dict__ for run in raw_data], f, indent=2)
        return raw_data

    def process_data(self, raw_data):
        processed_data = []
        for run in raw_data:
            mce = self._convert_run_to_mce(run)
            processed_data.append(mce)
        if self.save_debug_data and self.processing_dir:
            processed_data_path = self.processing_dir / 'mce_output.json'
            with open(processed_data_path, 'w') as f:
                json.dump(processed_data, f, indent=2)
        return processed_data

    def emit_data(self, processed_data):
        # Emit processed data, e.g., save to file or send to DataHub
        for mce in processed_data:
            # Implement emission logic
            pass  # Replace with actual emission code

    def _convert_run_to_mce(self, run):
        # Implement the conversion logic from LLMRun to MCE
        mce = {
            "id": run.id,
            "metrics": run.metrics,
            "metadata": run.metadata,
            # Add additional fields as needed
        }
        return mce
