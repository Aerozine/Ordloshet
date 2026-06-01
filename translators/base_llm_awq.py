"""Base class for pre-quantized AWQ LLM adapters."""
from __future__ import annotations

from .base_llm import LLMBase


class LLMBaseAWQ(LLMBase):
    """Loads a pre-quantized AWQ checkpoint via the autoawq library.

    Requires: pip install autoawq
    Set MODEL_ID to any HuggingFace AWQ checkpoint (e.g. Qwen/Qwen2.5-7B-Instruct-AWQ).
    AWQ models use fused kernels (fuse_layers=True) for ~1.5x throughput vs bitsandbytes Q4.
    """

    def _load(self):
        from transformers import AutoTokenizer

        try:
            from awq import AutoAWQForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "autoawq is required for AWQ adapters. Run: pip install autoawq"
            ) from exc

        self._suppress_hf_logging()
        print(f"\nLoading {self.LABEL} (AWQ 4-bit) from {self.MODEL_ID}...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoAWQForCausalLM.from_quantized(
            self.MODEL_ID,
            fuse_layers=True,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            device_map="auto",
        )
        size = f" ({self.MODEL_SIZE_MSG})" if self.MODEL_SIZE_MSG else ""
        print(f"AWQ model loaded{size}")
        return model, tokenizer

    def _input_device(self):
        import torch
        # AWQ fused model may not expose standard .parameters(); fall back to CUDA if available.
        try:
            return next(self._model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
