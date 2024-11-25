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
from src.utils.model_utils import get_capabilities_from_model, get_provider_from_model
from src.emitters.json_emitter import JSONEmitter
from src.emitters.datahub import DataHubEmitter
from datahub.metadata.schema_classes import (
    MetadataChangeEventClass,
    DatasetSnapshotClass,
    DatasetPropertiesClass,
    MLModelSnapshotClass,
    MLModelPropertiesClass,
    # Include custom aspect classes here if available
)

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
            provider=get_provider_from_model(model_name),
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
    def __init__(self, config, save_debug_data=True, processing_dir=None,
                 emit_to_datahub=True, datahub_emitter=None):
        self.connector = LangSmithConnector(config)
        self.save_debug_data = save_debug_data
        self.processing_dir = Path(processing_dir) if processing_dir else None
        self.emit_to_datahub = emit_to_datahub
        self.datahub_emitter = datahub_emitter

        # Initialize JSONEmitter if needed
        if self.save_debug_data and self.processing_dir:
            self.json_emitter = JSONEmitter(self.processing_dir)
        else:
            self.json_emitter = None

    def fetch_data(self):
        """Fetch data and save debug output if enabled"""
        raw_data = self.connector.get_runs()

        if self.save_debug_data and self.processing_dir:
            try:
                # Ensure processing directory exists
                self.processing_dir.mkdir(parents=True, exist_ok=True)

                raw_data_path = self.processing_dir / 'langsmith_api_output.json'

                # Convert runs to serializable format with better error handling
                serializable_data = []
                for run in raw_data:
                    try:
                        run_dict = {
                            'id': str(getattr(run, 'id', '')),
                            'start_time': getattr(run, 'start_time', '').isoformat() if getattr(run, 'start_time', None) else '',
                            'end_time': getattr(run, 'end_time', '').isoformat() if getattr(run, 'end_time', None) else '',
                            'metrics': dict(getattr(run, 'metrics', {})),
                            'inputs': dict(getattr(run, 'inputs', {})),
                            'outputs': dict(getattr(run, 'outputs', {})),
                            'metadata': dict(getattr(run, 'metadata', {})),
                            'error': str(getattr(run, 'error', '')),
                            'tags': list(getattr(run, 'tags', [])),
                            'feedback_stats': dict(getattr(run, 'feedback_stats', {}))
                        }
                        serializable_data.append(run_dict)
                    except Exception as e:
                        print(f"Warning: Failed to serialize run data: {e}")
                        continue

                # Write to file with proper encoding and error handling
                try:
                    with open(raw_data_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_data, f, indent=2, default=str)
                    print(f"Debug data saved to {raw_data_path}")
                except Exception as e:
                    print(f"Error writing debug file: {e}")

            except Exception as e:
                print(f"Warning: Failed to save debug data: {e}")

        return raw_data

    def process_data(self, raw_data):
        processed_data = []
        for run in raw_data:
            mce = self._convert_run_to_mce(run)
            processed_data.append(mce)
        if self.save_debug_data and self.processing_dir:
            processed_data_path = self.processing_dir / 'mce_output.json'
            with open(processed_data_path, 'w') as f:
                # Convert MCEs to serializable dictionaries
                json.dump([mce.to_obj() for mce in processed_data], f, indent=2)
        return processed_data

    def emit_data(self, processed_data):
        # Emit processed data to DataHub if enabled
        if self.emit_to_datahub and self.datahub_emitter:
            for mce in processed_data:
                try:
                    self.datahub_emitter.emit(mce)
                    print(f"Emitted MCE to DataHub: {mce.proposedSnapshot.urn}")
                except Exception as e:
                    print(f"Error emitting to DataHub: {e}")
        else:
            print("DataHub emission for runs is disabled or emitter not available.")

        # Emit processed data to JSON if enabled
        if self.json_emitter:
            # Convert MCEs to serializable dictionaries
            self.json_emitter.emit([mce.to_obj() for mce in processed_data], 'processed_data.json')

    def _convert_run_to_mce(self, run) -> MetadataChangeEventClass:
        """Convert LangSmith run to DataHub MCE with proper JSON handling"""
        try:
            # Helper function to safely convert values to JSON
            def safe_json_dumps(value, default=None):
                if value is None:
                    return json.dumps(default)
                try:
                    # Handle Mock objects by getting their return value
                    if hasattr(value, '_mock_return_value'):
                        value = value._mock_return_value
                    # Convert to dict if possible
                    if hasattr(value, '__dict__'):
                        value = value.__dict__
                    return json.dumps(value)
                except (TypeError, AttributeError):
                    return json.dumps(default)

            # Helper function to safely get datetime
            def safe_datetime(dt_value):
                if dt_value is None:
                    return ""
                if isinstance(dt_value, datetime):
                    return dt_value.isoformat()
                return ""

            # Safely extract and convert run attributes
            run_attributes = {
                "run_id": str(getattr(run, 'id', '')),
                "start_time": safe_datetime(getattr(run, 'start_time', None)),
                "end_time": safe_datetime(getattr(run, 'end_time', None)),
                "metrics": safe_json_dumps(getattr(run, 'metrics', {}), default={}),
                "inputs": safe_json_dumps(getattr(run, 'inputs', {}), default={}),
                "outputs": safe_json_dumps(getattr(run, 'outputs', {}), default={}),
                "metadata": safe_json_dumps(getattr(run, 'metadata', {}), default={})
            }

            # Create URN for the run
            run_urn = f"urn:li:dataset:(urn:li:dataPlatform:langsmith,runs/{run_attributes['run_id']},PROD)"

            # Create MCE
            return MetadataChangeEventClass(
                proposedSnapshot=DatasetSnapshotClass(
                    urn=run_urn,
                    aspects=[
                        DatasetPropertiesClass(
                            name=f"LLM Run {run_attributes['run_id']}",
                            description=f"Run ID: {run_attributes['run_id']}",
                            customProperties=run_attributes
                        )
                    ]
                )
            )
        except Exception as e:
            print(f"Error converting run to MCE: {e}")
            raise

    def process_models(self, models):
        # Process models if any additional processing is needed
        pass

    def emit_models(self, models):
        # Emit models to DataHub if enabled
        if self.emit_to_datahub and self.datahub_emitter:
            for model in models:
                try:
                    model_urn = f"urn:li:mlModel:(urn:li:dataPlatform:{model.provider},{model.name},PROD)"
                    mce = MetadataChangeEventClass(
                        proposedSnapshot=MLModelSnapshotClass(
                            urn=model_urn,
                            aspects=[
                                MLModelPropertiesClass(
                                    name=model.name,
                                    description=f"{model.provider} {model.name}",
                                    customProperties={
                                        "provider": model.provider,
                                        "model_family": model.model_family,
                                        "capabilities": json.dumps(model.capabilities),
                                        "parameters": json.dumps(model.parameters),
                                        "metadata": json.dumps(model.metadata)
                                    }
                                )
                            ]
                        )
                    )
                    self.datahub_emitter.emit(mce)
                    print(f"Emitted model to DataHub: {model.name}")
                except Exception as e:
                    print(f"Error emitting model to DataHub: {e}")
        else:
            print("DataHub emission for models is disabled or emitter not available.")

        # Save models to JSON if enabled
        if self.json_emitter:
            self.json_emitter.emit([model.to_dict() for model in models], 'models.json')
