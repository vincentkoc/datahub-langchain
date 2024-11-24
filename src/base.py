from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

@dataclass
class LLMModel:
    """Represents an LLM model"""
    name: str
    provider: str
    model_family: str
    capabilities: List[str]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class LLMRun:
    """Represents a single LLM execution"""
    id: str
    start_time: datetime
    end_time: Optional[datetime]
    model: Optional[LLMModel]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metrics: Dict[str, Any]
    parent_id: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class LLMChain:
    """Represents an LLM chain/pipeline"""
    id: str
    name: str
    components: List[str]
    config: Dict[str, Any]
    metadata: Dict[str, Any]

class LLMPlatformConnector(ABC):
    """Base interface for connecting to LLM platforms"""

    @abstractmethod
    def get_models(self) -> List[LLMModel]:
        """Get available models"""
        pass

    @abstractmethod
    def get_runs(self, **filters) -> List[LLMRun]:
        """Get run history"""
        pass

    @abstractmethod
    def get_chains(self) -> List[LLMChain]:
        """Get chain definitions"""
        pass

class LLMObserver(ABC):
    """Base interface for LLM observation"""

    @abstractmethod
    def start_run(self, run_id: str, **kwargs) -> None:
        """Start observing a run"""
        pass

    @abstractmethod
    def end_run(self, run_id: str, **kwargs) -> None:
        """End observing a run"""
        pass

    @abstractmethod
    def log_metrics(self, run_id: str, metrics: Dict[str, Any]) -> None:
        """Log metrics for a run"""
        pass

    @abstractmethod
    def log_model(self, model: LLMModel) -> None:
        """Log model metadata"""
        pass

    @abstractmethod
    def log_chain(self, chain: LLMChain) -> None:
        """Log chain metadata"""
        pass

class LLMMetadataEmitter(ABC):
    """Base interface for emitting LLM metadata"""

    @abstractmethod
    def emit_model(self, model: LLMModel) -> str:
        """Emit model metadata"""
        pass

    @abstractmethod
    def emit_run(self, run: LLMRun) -> str:
        """Emit run metadata"""
        pass

    @abstractmethod
    def emit_chain(self, chain: LLMChain) -> str:
        """Emit chain metadata"""
        pass

    @abstractmethod
    def emit_lineage(self, source_urn: str, target_urn: str, lineage_type: str) -> None:
        """Emit lineage between entities"""
        pass
