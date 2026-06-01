"""MADLAD-400 10B with speculative decoding via MADLAD-3B as assistant.

VRAM requirements: ~5GB (10B NF4) + ~6GB (3B fp16) = ~11GB. Tight on a 4070 (12GB).
Reduces latency ~1.5x with identical output quality.
"""
from __future__ import annotations

from .base_seq2seq import Seq2SeqBase


class _MADLADSpec(Seq2SeqBase):
    MODEL_ID = "google/madlad400-10b-mt"
    ASSISTANT_MODEL_ID = "google/madlad400-3b-mt"
    LABEL = "MADLAD-Spec"
    SOURCE_PREFIX = "<2fr> "
    MODEL_SIZE_MSG = "~11GB combined (10B NF4 + 3B fp16)"

    def __init__(self) -> None:
        super().__init__()
        self._assistant_model = None

    def _load(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
        from .translatorOptim import apply_torch_compile, from_pretrained_with_attention

        self._suppress_hf_logging()

        if not torch.cuda.is_available():
            raise RuntimeError("Speculative decoding requires a CUDA GPU.")

        print(f"\nLoading {self.LABEL}: main model (10B NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        model = from_pretrained_with_attention(
            AutoModelForSeq2SeqLM,
            self.MODEL_ID,
            self.LABEL,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        print(f"Loading {self.LABEL}: assistant model (3B fp16)...")
        assistant = AutoModelForSeq2SeqLM.from_pretrained(
            self.ASSISTANT_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        assistant.eval()
        self._assistant_model = assistant
        print(f"Speculative decoding ready ({self.MODEL_SIZE_MSG}).")
        return model, tokenizer

    def _gen_kwargs(self) -> dict:
        kwargs = super()._gen_kwargs()
        if self._assistant_model is not None:
            kwargs["assistant_model"] = self._assistant_model
        return kwargs

    def unload_model(self) -> None:
        self._assistant_model = None
        super().unload_model()


_inst = _MADLADSpec()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
translate_many = _inst.translate_many
count_tokens_many = _inst.count_tokens_many
unload_model = _inst.unload_model
largetranslate = large_translate
