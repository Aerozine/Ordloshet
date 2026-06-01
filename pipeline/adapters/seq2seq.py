"""pipeline/adapters/seq2seq.py  -  Seq2Seq adapter for NLLB/MADLAD/OPUS."""


class Seq2SeqAdapter:
    """Adapter for seq2seq models (NLLB, MADLAD, OPUS)."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path.lower()
    
    @property
    def memory_usage_gb(self) -> float:
        """Estimate VRAM usage for seq2seq models."""
        if 'nllb' in self.model_path and '3.3b' not in self.model_path:
            return 6.5  # NLLB-200 base ~6.5GB
        elif 'madlad' in self.model_path or '400-10b' in self.model_path:
            return 6.0  # MADLAD ~6GB
        elif 'opus' in self.model_path:
            return 2.0  # OPUS-MT ~2GB
        return 1.5
    
    def translate(self, text: str) -> str:
        """Translate using seq2seq model."""
        # Fallback stub - actual translation happens in translatorNLLB.py/large_translate()
        return f"[Translated {text}]"


__all__ = ['Seq2SeqAdapter']
