"""MADLAD-400 10B MT adapter with 4-bit NF4 quantization (bitsandbytes)."""
from .base_seq2seq import Seq2SeqBase


class _MADLADQ4(Seq2SeqBase):
    MODEL_ID = "google/madlad400-10b-mt"
    LABEL = "MADLAD-Q4"
    SOURCE_PREFIX = "<2fr> "
    MODEL_SIZE_MSG = "~5GB (4-bit NF4)"

    def _load_one(self, model_id: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
        from .translatorOptim import apply_torch_compile, from_pretrained_with_attention

        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes is required for 4-bit quantization. "
                "Run: pip install bitsandbytes"
            ) from exc

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = from_pretrained_with_attention(
            AutoModelForSeq2SeqLM,
            model_id,
            self.LABEL,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        model = apply_torch_compile(model, self.LABEL)
        return model, tokenizer


_inst = _MADLADQ4()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
translate_many = _inst.translate_many
count_tokens_many = _inst.count_tokens_many
unload_model = _inst.unload_model
largetranslate = large_translate
