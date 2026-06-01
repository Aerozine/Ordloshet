"""ALMA-7B-R literary translation adapter (haoranxu/ALMA-7B-R).

ALMA (Advanced Language Model-based trAnslation) is specifically RLHF-fine-tuned
for high-quality translation and consistently outperforms NLLB/MADLAD on literary
benchmarks. Loaded in NF4 4-bit quantization (~4GB) to fit within 12GB VRAM.

Prompt format follows the ALMA training convention:
    Translate this from English to French:
    English: {source}
    French:
"""
from __future__ import annotations

from .base_llm import LLMBase

LABEL = "ALMA-7B-R"


class _ALMA(LLMBase):
    MODEL_ID = "haoranxu/ALMA-7B-R"
    LABEL = LABEL
    MODEL_SIZE_MSG = "~4GB (NF4 4-bit)"
    USE_BFLOAT16 = True
    MAX_INPUT_TOKENS = 2048
    TRANSLATE_SYSTEM = None  # ALMA uses no system message — only the instruction prefix

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from .translatorOptim import apply_torch_compile

        self._suppress_hf_logging()

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\nLoading {self.LABEL} (NF4 4-bit)... GPU: {gpu_name}, VRAM: {total_gb:.2f}GB")
        else:
            print(f"\nLoading {self.LABEL}... WARNING: No CUDA GPU detected, CPU mode.")

        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.TRUST_REMOTE_CODE,
        )
        model.eval()
        model = apply_torch_compile(model, self.LABEL)
        print(f"ALMA-7B-R loaded ({self.MODEL_SIZE_MSG})")
        return model, tokenizer

    def _alma_prompt(self, text: str) -> str:
        """Build the ALMA translation prompt in LLaMA-2 instruction format."""
        user_msg = (
            f"Translate this from English to French:\n"
            f"English: {text.strip()}\n"
            "French:"
        )
        return self._chat_prompt(user_msg, system_text=None)

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        self.ensure_model_loaded()
        prompt = self._alma_prompt(text)
        return self._generate_from_prompt(prompt)

    def translate_many(self, texts: list[str], batch_size: int = 4) -> list[str]:
        if not texts:
            return []
        self.ensure_model_loaded()
        return [self.translate(t) for t in texts]


_inst = _ALMA()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
translate_many = _inst.translate_many
build_chapter_memo = _inst.build_chapter_memo
patch_translation = _inst.patch_translation
repair_translation = _inst.repair_translation
unload_model = _inst.unload_model
largetranslate = large_translate
