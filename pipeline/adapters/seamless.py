"""pipeline/adapters/seamless.py  -  SeamlessM4T adapter.

Implements multi-modal translation (speech/text/image to French).
Currently supports text-to-text mode only.
"""
from typing import Dict, Optional


class SeamlessAdapter:
    """
    Adapter for SeamlessM4T model.
    
    Multi-modal model that supports speech, text, and image inputs.
    Currently used for text-to-text translation.
    """
    
    def __init__(self, model_name: str = 'facebook/seamless-m4t-turbo'):
        self.model_name = model_name
    
    @property
    def model_name(self) -> str:
        return self._model_name if hasattr(self, '_model_name') else 'unknown'
    
    @model_name.setter
    def model_name(self, name: str):
        self._model_name = name
    
    @property
    def memory_usage_gb(self) -> float:
        """Estimate VRAM usage for SeamlessM4T."""
        # SeamlessM4T-large is ~6.5GB, turbo is ~3GB
        if 'turbo' in self.model_name.lower():
            return 3.0
        return 6.5
    
    def initialize(self):
        """Load model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            dtype = torch.float16 if hasattr(torch, 'float16') else torch.float32
            from transformers import AutoModelForSeq2SeqLM
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
            
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {e}")
    
    def translate(self, text: str) -> str:
        """Translate using SeamlessM4T model."""
        if not hasattr(self, 'model'):
            self.initialize()
        
        try:
            from torch import no_grad
            
            device = next(p.device for p in self.model.parameters())
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)
            
            with no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                )
            
            return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"SeamlessM4T translation error: {e}")
            raise
