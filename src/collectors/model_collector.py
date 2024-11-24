from typing import List, Dict, Any
from ..base import LLMModel, LLMPlatformConnector

class ModelCollector:
    """Collects model information from various platforms"""

    def __init__(self, connectors: List[LLMPlatformConnector]):
        self.connectors = connectors

    def collect_models(self) -> List[LLMModel]:
        """Collect models from all configured platforms"""
        models = []
        for connector in self.connectors:
            try:
                platform_models = connector.get_models()
                models.extend(platform_models)
            except Exception as e:
                print(f"Error collecting models from {connector.__class__.__name__}: {e}")
        return models

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about collected models"""
        models = self.collect_models()
        return {
            "total_models": len(models),
            "by_provider": self._group_by_provider(models),
            "by_capability": self._group_by_capability(models)
        }

    def _group_by_provider(self, models: List[LLMModel]) -> Dict[str, int]:
        provider_counts = {}
        for model in models:
            provider_counts[model.provider] = provider_counts.get(model.provider, 0) + 1
        return provider_counts

    def _group_by_capability(self, models: List[LLMModel]) -> Dict[str, int]:
        capability_counts = {}
        for model in models:
            for capability in model.capabilities:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
        return capability_counts
