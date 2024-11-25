from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import uuid
from uuid import UUID

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
from src.utils.model_utils import (
    get_capabilities_from_model,
    get_provider_from_model,
    get_model_family,
    get_model_parameters
)
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
        """Get run history from LangSmith with smart aggregation by trace"""
        try:
            if not self.client:
                print("\n=== DEBUG: Client Status ===")
                print("Client is None, attempting to initialize...")
                try:
                    self.client = Client(
                        api_url=self.config.get_platform_config("langsmith")["endpoint"],
                        api_key=self.config.get_platform_config("langsmith")["api_key"]
                    )
                    print("✓ Client initialized successfully")
                except Exception as e:
                    print(f"✗ Failed to initialize LangSmith client: {e}")
                    return []

            if start_time is None:
                start_time = datetime.now() - timedelta(days=self.config.ingest_window_days)

            print("\n=== DEBUG: API Call Parameters ===")
            print(f"Project: {self.project_name}")
            print(f"Start time: {start_time}")

            try:
                print("\n=== DEBUG: Making API Call ===")
                # Get all runs within window
                runs_iterator = self.client.list_runs(
                    project_name=self.project_name,
                    start_time=start_time,
                    end_time=end_time
                )
                print("✓ API call successful")

                # Convert iterator to list safely with debug info
                all_runs = []
                run_count = 0
                for run in runs_iterator:
                    run_count += 1
                    if run_count == 1:  # Debug first run in detail
                        print("\nFirst Run Details:")
                        print(f"Run ID: {getattr(run, 'id', 'unknown')}")
                        print(f"Run Type: {getattr(run, 'run_type', 'unknown')}")
                        print(f"Extra: {getattr(run, 'extra', {})}")

                    if run is not None:
                        all_runs.append(run)

                print(f"\nFound {len(all_runs)} runs")

                if not all_runs:
                    print("⚠️ No runs found")
                    return []

                # Process runs into LLMRun objects
                processed_runs = []
                for idx, run in enumerate(all_runs):
                    try:
                        # Extract model info from extra field
                        model_info = None
                        extra = getattr(run, 'extra', {}) or {}
                        if extra and isinstance(extra, dict):
                            invocation_params = extra.get('invocation_params', {})
                            if invocation_params:
                                model_info = {
                                    'model_name': invocation_params.get('model_name', 'unknown'),
                                    'temperature': invocation_params.get('temperature', 0.7),
                                    'max_tokens': invocation_params.get('max_tokens'),
                                    **extra.get('metadata', {})
                                }

                        # Create LLMModel if we have model info
                        model = None
                        if model_info:
                            model = LLMModel(
                                name=model_info.get('model_name', 'unknown'),
                                provider=get_provider_from_model(model_info.get('model_name', '')),
                                model_family=get_model_family(model_info.get('model_name', '')),
                                capabilities=get_capabilities_from_model(model_info.get('model_name', '')),
                                parameters=model_info,
                                metadata={
                                    "source": "langsmith",
                                    "project": self.project_name,
                                    "runtime": model_info.get('runtime'),
                                    "platform": model_info.get('platform')
                                }
                            )

                        # Get run metrics
                        metrics = {
                            "latency": getattr(run, 'latency', 0) or 0,
                            "token_usage": getattr(run, 'token_usage', {}) or {},
                            "total_tokens": getattr(run, 'total_tokens', 0) or 0,
                            "prompt_tokens": getattr(run, 'prompt_tokens', 0) or 0,
                            "completion_tokens": getattr(run, 'completion_tokens', 0) or 0,
                            "total_cost": getattr(run, 'total_cost', 0) or 0,
                            "error_rate": 1.0 if getattr(run, 'error', None) else 0.0,
                            "success_rate": 0.0 if getattr(run, 'error', None) else 1.0
                        }

                        # Create LLMRun
                        llm_run = LLMRun(
                            id=str(getattr(run, 'id', str(uuid.uuid4()))),
                            start_time=getattr(run, 'start_time', datetime.now()),
                            end_time=getattr(run, 'end_time', datetime.now()),
                            model=model,
                            inputs=getattr(run, 'inputs', {}) or {},
                            outputs=getattr(run, 'outputs', {}) or {},
                            metrics=metrics,
                            parent_id=str(getattr(run, 'parent_run_id', '')) or None,
                            metadata={
                                "source": "langsmith",
                                "project": self.project_name,
                                "run_type": getattr(run, 'run_type', 'unknown'),
                                "trace_id": str(getattr(run, 'trace_id', '')) or '',
                                "error": str(getattr(run, 'error', '')) or '',
                                "tags": list(getattr(run, 'tags', [])) or [],
                                "extra": extra
                            }
                        )

                        processed_runs.append(llm_run)

                    except Exception as e:
                        print(f"Error creating LLMRun for {getattr(run, 'id', 'unknown')}: {e}")
                        if idx == 0:  # Only show detailed error for first failure
                            print("\n=== DEBUG: Error Details ===")
                            print(f"Run attributes: {dir(run)}")
                            print(f"Run dict: {run.__dict__ if hasattr(run, '__dict__') else 'No __dict__'}")
                            import traceback
                            print(f"Traceback: {traceback.format_exc()}")
                        continue

                print(f"\n=== DEBUG: Final Summary ===")
                print(f"Successfully processed {len(processed_runs)} out of {len(all_runs)} runs")

                return processed_runs

            except Exception as e:
                print(f"Error fetching runs from LangSmith API: {e}")
                if self.config.langchain_verbose:
                    import traceback
                    print(f"\n=== DEBUG: API Error Details ===")
                    print(traceback.format_exc())
                return []

        except Exception as e:
            print(f"Error in get_runs: {e}")
            if self.config.langchain_verbose:
                import traceback
                print(f"\n=== DEBUG: General Error Details ===")
                print(traceback.format_exc())
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

    def _convert_run(self, run) -> Optional[LLMRun]:
        """Convert LangSmith run to LLMRun with error handling"""
        try:
            # Extract model info if available
            model = None
            if hasattr(run, 'execution_metadata') and run.execution_metadata:
                model_name = run.execution_metadata.get('model_name')
                if model_name:
                    model = LLMModel(
                        name=model_name,
                        provider=get_provider_from_model(model_name),
                        model_family=get_model_family(model_name),
                        capabilities=get_capabilities_from_model(model_name),
                        parameters={},
                        metadata={
                            "source": "langsmith",
                            "project": self.project_name
                        }
                    )

            # Extract metrics with defaults
            metrics = {
                "latency": getattr(run, 'latency', 0),
                "token_usage": getattr(run, 'token_usage', {}),
                "cost": getattr(run, 'cost', 0),
                "error_rate": 1.0 if getattr(run, 'error', None) else 0.0,
                "success_rate": 0.0 if getattr(run, 'error', None) else 1.0
            }

            # Create LLMRun object
            return LLMRun(
                id=str(getattr(run, 'id', '')),
                start_time=getattr(run, 'start_time', datetime.now()),
                end_time=getattr(run, 'end_time', datetime.now()),
                model=model,
                inputs=dict(getattr(run, 'inputs', {})),
                outputs=dict(getattr(run, 'outputs', {})),
                metrics=metrics,
                metadata={
                    "error": str(getattr(run, 'error', '')),
                    "tags": list(getattr(run, 'tags', [])),
                    "feedback_stats": dict(getattr(run, 'feedback_stats', {})),
                    "source": "langsmith",
                    "project": self.project_name,
                    "run_type": getattr(run, 'run_type', 'unknown')
                }
            )
        except Exception as e:
            print(f"Error converting run {getattr(run, 'id', 'unknown')}: {e}")
            return None

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
        """Convert LangSmith run to DataHub MCE"""
        try:
            # First emit model if present
            model_urn = None
            if run.model:
                model_urn = f"urn:li:mlModel:(urn:li:dataPlatform:langsmith,{run.model.name},PROD)"

            # Create pipeline URN using project name as the main identifier
            pipeline_urn = f"urn:li:mlModel:(urn:li:dataPlatform:langsmith,{self.connector.project_name},PROD)"

            # Aggregate run data into the pipeline properties
            custom_properties = {
                "project": str(self.connector.project_name),
                "model_urn": str(model_urn) if model_urn else "",
                "latest_run_id": str(run.id),
                "latest_run_time": run.start_time.isoformat() if run.start_time else "",
                "latest_run_status": "success" if not run.metrics.get("error") else "failed",
                "total_runs": "1",  # Could be incremented if we track state
                "metrics": str({
                    "latency": run.metrics.get("latency", 0),
                    "token_usage": run.metrics.get("token_usage", {}),
                    "cost": run.metrics.get("cost", 0),
                    "error_rate": run.metrics.get("error_rate", 0.0),
                    "success_rate": run.metrics.get("success_rate", 1.0)
                }),
                "run_history": str([{  # Keep last N runs as history
                    "id": run.id,
                    "time": run.start_time.isoformat() if run.start_time else "",
                    "status": "success" if not run.metrics.get("error") else "failed",
                    "metrics": run.metrics
                }])
            }

            # Create run properties using MLModelProperties
            properties = MLModelPropertiesClass(
                description=f"LangSmith Project: {self.connector.project_name}",
                type="LangSmith Pipeline",  # Identify as a LangSmith pipeline
                customProperties=custom_properties
            )

            # Create MCE with MLModelSnapshot
            return MetadataChangeEventClass(
                proposedSnapshot=MLModelSnapshotClass(
                    urn=pipeline_urn,
                    aspects=[properties]
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
