"""pipeline/adapters package exports."""
from .base import AbstractTranslator


# Model name mappings - use full HuggingFace repo IDs that work
MODEL_REGISTRY = {
    "nllb": "facebook/nllb-200-3.3B",  # No Language Left Behind
    "madlad": "google/madlad400-10b-mt",  # Multilingual Audio Language Model  
    "opus": "Helsinki-NLP/opus-mt-tc-big-en-fr",  # OPUS-MT baseline
    "gemma": "google/translategemma-12b-it",  # Google Translate Gemma 12B
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",  # Mistral 7B Instruct
}


def get_adapter(model_name: str, strategy: str = "light") -> object:
    """Factory function to get adapter based on model and strategy."""
    
    # Map short name to full HuggingFace repo ID
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Use one of: {list(MODEL_REGISTRY.keys())}"
        )
    
    # Import here to avoid circular imports
    from .seq2seq import Seq2SeqAdapter
    from .llm import LLMAdapter
    
    # Determine adapter type based on model name
    if model_name in ["nllb", "madlad", "opus"]:
        return Seq2SeqAdapter(MODEL_REGISTRY[model_name])
    elif model_name in ["gemma", "mistral"]:
        return LLMAdapter(MODEL_REGISTRY[model_name], strategy=strategy)
    else:
        raise ValueError(f"Unknown adapter type for model: {model_name}")


__all__ = ['AbstractTranslator', 'get_adapter']
