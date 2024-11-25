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
