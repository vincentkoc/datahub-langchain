import re
from typing import List, Any, Dict, Optional

def normalize_model_name(model_name: str) -> str:
    """Normalize model name to standard format"""
    if not model_name or model_name == "unknown":
        return "unknown_model"

    # Remove any whitespace and convert to lowercase
    model_name = model_name.strip().lower()

    # Handle common prefixes
    prefixes = ["openai/", "anthropic/", "meta/", "google/"]
    for prefix in prefixes:
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix):]

    # Handle class names vs actual model names
    if model_name == "chatopenai":
        return "gpt-3.5-turbo"  # Default OpenAI chat model

    return model_name

def get_provider_from_model(model_name: str) -> str:
    """Get provider from model name"""
    model_name = model_name.lower()

    if "gpt" in model_name or model_name.startswith("text-") or "openai" in model_name:
        return "OpenAI"
    elif "claude" in model_name:
        return "Anthropic"
    elif "llama" in model_name:
        return "Meta"
    elif "palm" in model_name or "gemini" in model_name:
        return "Google"
    return "Unknown"

def get_model_family(model_name: str) -> str:
    """Get model family from name"""
    model_name = model_name.lower()

    # First check for specific model families
    if "gpt-4" in model_name:
        return "GPT-4"
    elif "gpt-3.5" in model_name or model_name == "chatopenai":  # Add class name mapping
        return "GPT-3.5"
    elif "claude" in model_name:
        if "2" in model_name:
            return "Claude-2"
        return "Claude"
    elif "llama" in model_name:
        if "2" in model_name:
            return "LLaMA-2"
        return "LLaMA"
    elif "palm" in model_name:
        return "PaLM"
    elif "gemini" in model_name:
        return "Gemini"

    # If no specific family found, try to extract from class name
    if "openai" in model_name:
        return "GPT-3.5"  # Default for OpenAI

    return "Unknown"

def get_capabilities_from_model(model_name: str) -> List[str]:
    """Get model capabilities based on name"""
    capabilities = ["text-generation"]

    model_name = model_name.lower()
    if "vision" in model_name or "-v" in model_name:
        capabilities.append("image-understanding")
    if "audio" in model_name or "whisper" in model_name:
        capabilities.append("speech-to-text")
    if "embedding" in model_name:
        capabilities.append("embeddings")
    if "instruct" in model_name:
        capabilities.append("instruction-following")
    if "code" in model_name or "codex" in model_name:
        capabilities.append("code-generation")

    return capabilities

def get_model_parameters(model: Any) -> Dict[str, Any]:
    """Get model parameters with better defaults"""
    params = {}

    # Common parameters to look for
    param_names = [
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "model_name",
        "model_kwargs"
    ]

    # Extract from dict or object
    if isinstance(model, dict):
        for param in param_names:
            if param in model:
                params[param] = model[param]
    else:
        for param in param_names:
            if hasattr(model, param):
                value = getattr(model, param)
                if value is not None:
                    params[param] = value

    # Add model-specific defaults if not present
    model_name = (
        model.get("model_name", None) if isinstance(model, dict)
        else getattr(model, "model_name", None)
    )

    if model_name:
        if "gpt-4" in str(model_name):
            params.setdefault("contextWindow", 8192)
            params.setdefault("tokenLimit", 8192)
            params.setdefault("costPerToken", 0.0003)
        elif "gpt-3.5" in str(model_name):
            params.setdefault("contextWindow", 4096)
            params.setdefault("tokenLimit", 4096)
            params.setdefault("costPerToken", 0.0001)
        elif "claude" in str(model_name):
            params.setdefault("contextWindow", 100000)
            params.setdefault("tokenLimit", 100000)
            params.setdefault("costPerToken", 0.0001)

    return params
