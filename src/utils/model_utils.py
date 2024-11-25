import re
from typing import List

def get_capabilities_from_model(model_name: str) -> List[str]:
    """Determine model capabilities from model name using regex."""
    capabilities = ['text-generation']
    model_name_lower = model_name.lower()

    # Define regex patterns for matching capabilities
    if re.search(r'(gpt-4|gpt-3\.5|gpt-)', model_name_lower):
        capabilities.append('chat')
        capabilities.append('function-calling')
    # Add more patterns as needed

    return capabilities

def get_provider_from_model(model_name: str) -> str:
    """Determine provider from model name using regex."""
    model_name_lower = model_name.lower()

    if 'gpt-' in model_name_lower or 'openai' in model_name_lower:
        return 'OpenAI'
    elif 'claude' in model_name_lower or 'anthropic' in model_name_lower:
        return 'Anthropic'
    # Add more provider detections as needed
    else:
        return 'Unknown'
