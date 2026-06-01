"""pipeline/translator.py  -  Translator registry and abstract interface.

Provides a unified interface for all translators with auto-loading from models/registry.toml.
Supports both seq2seq (NLLB, MADLAD, OPUS) and LLM-based (Mistral, Gemma) models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import warnings


warnings.filterwarnings('ignore')


# Model registry configuration
def _load_registry() -> Dict:
    """Load model configurations from models/registry.toml."""
    try:
        with open('models/registry.toml', 'rb') as f:
            import tomli
            return tomli.load(f)
    except ImportError:
        # Fallback to basic dict if tomli not installed
        return {}


# Abstract Translator Interface
class AbstractTranslator(ABC):
    @abstractmethod
    def translate(self, text: str, src_lang: str = 'en', tgt_lang: str = 'fr') -> str:
        """Translate a text segment."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name for logging and VRAM tracking."""
        pass
    
    @property
    @abstractmethod
    def memory_usage_gb(self) -> float:
        """Estimated memory usage in GB (peak)."""
        pass


# Seq2Seq Translator Adapter
class Seq2SeqTranslator(AbstractTranslator):
    def __init__(self, model_name: str, config: Dict):
        self._model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialized = False

    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def memory_usage_gb(self) -> float:
        """Estimate VRAM usage (approximation)."""
        # Approximate: seq2seq models ~0.1GB per 1B parameters + overhead
        params_b = self.model_name.replace('nllb-', '0').replace('opus-', '0').replace('madlad', '0')
        if '10b' in self.model_name.lower():
            return 6.5  # NLLB-3.3B ~6.6GB, MADLAD-400-10B ~6GB
        elif '7b' in self.model_name.lower():
            return 4.5
        else:
            return 1.5
    
    def initialize(self):
        """Load model and tokenizer (lazy initialization)."""
        if self._initialized:
            return
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            dtype = torch.float16 if hasattr(torch, 'float16') else torch.float32
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",  # Auto-select GPU/CPU based on availability
            )
            
            self.model.eval()
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {e}")
    
    def translate(self, text: str) -> str:
        """Translate using seq2seq model."""
        if not self._initialized:
            self.initialize()

        import torch
        
        device = next(p.device for p in self.model.parameters())
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=8,
                    length_penalty=1.0,
                    early_stopping=True,
                )
            
            return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Seq2Seq translation error (retrying with CPU fallback): {e}")
            
            # Fallback to CPU if CUDA fails
            self.model = self.model.cpu()
            return self.translate(text)


# LLM Translator Adapter (Mistral, Gemma)
class LLMTranslator(AbstractTranslator):
    def __init__(self, model_name: str, config: Dict):
        self._model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialized = False

    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def memory_usage_gb(self) -> float:
        """Estimate VRAM usage for LLM models."""
        if '7b' in self.model_name.lower():
            return 4.5  # Mistral-7B 4-bit ~4.5GB
        elif '12b' in self.model_name.lower() or 'translate' in self.model_name.lower():
            return 6.0  # Gemma 4-bit ~6GB (quantized)
        else:
            return 8.0  # Assume larger model
    
    def initialize(self):
        """Load model and tokenizer with quantization."""
        if self._initialized:
            return
        
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        self.model.eval()
        self._initialized = True
    
    def translate(self, text: str) -> str:
        """Translate using LLM model with system prompt for register."""
        if not self._initialized:
            self.initialize()

        import torch
        
        device = next(p.device for p in self.model.parameters())
        
        # System prompt handles register (tu/vous) and tag preservation
        system_prompt = text  # Will be wrapped by pipeline
        
        messages = [
            {"role": "system", "content": f"Translate to French. Preserve style: {system_prompt}"},
            {"role": "user", "content": system_prompt},
        ]
        
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            new_tokens = output[0][prompt_len:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"LLM translation error (retrying): {e}")
            raise


# Translator Registry
class TranslatorRegistry:
    def __init__(self):
        self.models = {}  # model_name -> config
    
    def load(self, registry_path: str = 'models/registry.toml'):
        """Load model configs from TOML file."""
        self.models = _load_registry()
    
    def get(self, model_name: str) -> Optional[AbstractTranslator]:
        """Get translator instance for model name."""
        if model_name.lower() == 'nllb':
            return Seq2SeqTranslator(
                model_name='facebook/nllb-200-3.3B',
                config={'quantization': 'fp16'},
            )
        elif model_name.lower() == 'madlad':
            return Seq2SeqTranslator(
                model_name='google/madlad400-10b-mt',
                config={'quantization': 'nf4'},
            )
        elif model_name.lower() == 'opus':
            return Seq2SeqTranslator(
                model_name='Helsinki-NLP/opus-mt-tc-big-en-fr',
                config={'quantization': 'fp16'},
            )
        elif model_name.lower() in ['gemma', 'translate_gemma']:
            self.models['gemma'] = LLMTranslator(
                model_name='google/translategemma-12b-it',
                config={'quantization': 'nf4'},
            )
            return self.models['gemma']
        elif model_name.lower() == 'mistral':
            return LLMTranslator(
                model_name='mistralai/Mistral-7B-Instruct-v0.3',
                config={'quantization': 'nf4'},
            )
        else:
            return None


# Public API
def get_translator(model_name: str) -> Optional[AbstractTranslator]:
    """Get translator for model name (lazy initialization)."""
    registry = TranslatorRegistry()
    return registry.get(model_name)
