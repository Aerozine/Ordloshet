"""pipeline/adapters/llm.py  -  LLM adapter for Mistral/Gemma."""


class LLMAdapter:
    """Adapter for causal LLM models (Mistral, Gemma)."""
    
    def __init__(self, model_path: str, strategy: str = "light"):
        self.model_path = model_path.lower()
        self.strategy = strategy
    
    @property
    def memory_usage_gb(self) -> float:
        """Estimate VRAM usage for LLM models."""
        if 'gemma-12b' in self.model_path or 'translategemma' in self.model_path:
            return 6.0  # Gemma Translate ~6GB with quantization
        elif 'mistral-7b' in self.model_path:
            return 4.5  # Mistral ~4.5GB with quantization
        return 6.0
    
    def translate(self, text: str) -> str:
        """Translate using LLM model."""
        # Fallback stub - actual translation happens in translatorGemma.py/translate()
        return f"[Translated {text}]"


__all__ = ['LLMAdapter']
