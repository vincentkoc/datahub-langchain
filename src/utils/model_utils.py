import re
from typing import List, Any, Dict

def normalize_model_name(model_name: str) -> str:
    """Normalize model name to its canonical form."""
    if not model_name:
        return 'unknown'

    # Handle list/tuple model names
    if isinstance(model_name, (list, tuple)):
        model_name = model_name[-1] if model_name else 'unknown'

    # Extract full model name including version
    if match := re.search(r'(gpt-3\.5-turbo-\d+)', model_name.lower()):
        return match.group(1)
    elif match := re.search(r'(gpt-4-\d+)', model_name.lower()):
        return match.group(1)
    elif match := re.search(r'(gpt-3\.5-turbo|gpt-4|gpt-\d+)', model_name.lower()):
        return match.group(1)

    return model_name

def get_capabilities_from_model(model_name: str, model_instance: Any = None) -> List[str]:
    """Determine model capabilities from model name and instance using regex."""
    capabilities = ['text-generation']
    model_name_lower = model_name.lower()

    # Base capabilities from model name
    if re.search(r'(gpt-4|gpt-3\.5-turbo|gpt-)', model_name_lower):
        capabilities.append('chat')
        capabilities.append('function-calling')

    # Version-specific capabilities
    if re.search(r'(gpt-4-turbo|gpt-3\.5-turbo-0125)', model_name_lower):
        capabilities.append('json-mode')
        capabilities.append('reproducible-outputs')

    # Additional capabilities from model instance
    if model_instance:
        if isinstance(model_instance, dict):
            if model_instance.get('streaming', False):
                capabilities.append('streaming')
            if model_instance.get('functions'):
                capabilities.append('function-calling')
        else:
            if hasattr(model_instance, 'streaming') and model_instance.streaming:
                capabilities.append('streaming')
            if hasattr(model_instance, 'functions') and model_instance.functions:
                capabilities.append('function-calling')

    # Remove duplicates while preserving order
    return list(dict.fromkeys(capabilities))

def get_provider_from_model(model_name: str) -> str:
    """Determine provider from model name using regex."""
    model_name_lower = model_name.lower()

    if re.search(r'(gpt-|davinci|openai)', model_name_lower):
        return 'OpenAI'
    elif re.search(r'(claude|anthropic)', model_name_lower):
        return 'Anthropic'
    elif re.search(r'llama', model_name_lower):
        return 'Meta'
    elif re.search(r'palm|gemini', model_name_lower):
        return 'Google'
    else:
        return 'Unknown'

def get_model_family(model_name: str) -> str:
    """Determine model family from name using regex."""
    model_name_lower = model_name.lower()

    # Specific GPT-3.5 versions
    if re.search(r'gpt-3\.5-turbo-0125', model_name_lower):
        return 'GPT-3.5-Turbo-0125'
    elif re.search(r'gpt-3\.5-turbo-1106', model_name_lower):
        return 'GPT-3.5-Turbo-1106'
    elif re.search(r'gpt-3\.5-turbo-0613', model_name_lower):
        return 'GPT-3.5-Turbo-0613'
    # Specific GPT-4 versions
    elif re.search(r'gpt-4-0125-preview', model_name_lower):
        return 'GPT-4-0125-Preview'
    elif re.search(r'gpt-4-1106-preview', model_name_lower):
        return 'GPT-4-1106-Preview'
    # Generic families
    elif re.search(r'gpt-4', model_name_lower):
        return 'GPT-4'
    elif re.search(r'gpt-3\.5', model_name_lower):
        return 'GPT-3.5'
    elif re.search(r'claude', model_name_lower):
        return 'Claude'
    elif re.search(r'llama', model_name_lower):
        return 'LLaMA'
    elif re.search(r'palm', model_name_lower):
        return 'PaLM'
    elif re.search(r'gemini', model_name_lower):
        return 'Gemini'
    elif re.search(r'gpt-', model_name_lower):
        return 'GPT'
    else:
        return 'Language Model'

def get_model_parameters(model: Any) -> Dict[str, Any]:
    """Get model parameters with defaults."""
    params = {}

    # Model-specific defaults
    model_name = getattr(model, 'model_name', '').lower() if not isinstance(model, dict) else model.get('model_name', '').lower()

    if 'gpt-3.5-turbo-0125' in model_name:
        params.update({
            "contextWindow": 16385,
            "tokenLimit": 16385,
            "costPerToken": 0.0001
        })
    elif 'gpt-4' in model_name:
        params.update({
            "contextWindow": 8192,
            "tokenLimit": 8192,
            "costPerToken": 0.0003
        })
    else:
        params.update({
            "contextWindow": 4096,
            "tokenLimit": 4096,
            "costPerToken": 0.0001
        })

    # Standard LLM parameters
    param_names = [
        'temperature',
        'max_tokens',
        'top_p',
        'frequency_penalty',
        'presence_penalty'
    ]

    if isinstance(model, dict):
        for param in param_names:
            if param in model:
                params[param] = model[param]
    else:
        for param in param_names:
            if hasattr(model, param):
                params[param] = getattr(model, param)

    return params
