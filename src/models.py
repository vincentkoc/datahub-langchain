from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class Prompt:
    """Represents a prompt template"""
    template: str
    input_variables: List[str]
    template_format: str
    version: str
    metadata: Dict[str, Any]

@dataclass
class Tool:
    """Represents a tool/function"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class Metrics:
    """Common metrics for LLM operations"""
    latency: float
    token_usage: Dict[str, int]
    cost: float
    error_rate: float
    success_rate: float
    custom_metrics: Dict[str, Any]

@dataclass
class RAGComponent:
    """Represents a RAG component"""
    component_type: str  # "document_store", "vector_store", "retrieval_chain"
    config: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class LLMModel:
    # Define your attributes
    name: str
    provider: str
    model_family: str
    capabilities: List[str]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self):
        return {
            'name': self.name,
            'provider': self.provider,
            'model_family': self.model_family,
            'capabilities': self.capabilities,
            'parameters': self.parameters,
            'metadata': self.metadata,
        }

@dataclass
class LLMRun:
    # Define your attributes
    id: str
    start_time: datetime
    end_time: Optional[datetime]
    model: Optional[LLMModel]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metrics: Dict[str, Any]
    parent_id: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self):
        return {
            'id': self.id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'model': self.model.to_dict() if self.model else None,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'metrics': self.metrics,
            'parent_id': self.parent_id,
            'metadata': self.metadata,
        }
