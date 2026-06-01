"""Base class for HuggingFace seq2seq MT adapters (NLLB, MADLAD, OPUS)."""
from __future__ import annotations


class Seq2SeqBase:
    MODEL_ID: str = ""
    MODEL_IDS: list[str] = []   # if non-empty, tries each in order (OPUS fallback pattern)
    LABEL: str = ""
    SOURCE_PREFIX: str = ""     # prepended to source text (MADLAD: "<2fr> ")
    SRC_LANG: str = ""          # if set, passed to tokenizer src_lang (NLLB: "eng_Latn")
    FORCED_BOS_LANG: str = ""   # if set, used as forced_bos_token_id (NLLB: "fra_Latn")
    USE_BFLOAT16: bool = False
    MODEL_SIZE_MSG: str = ""
    MAX_LENGTH: int = 512

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def _suppress_hf_logging(self) -> None:
        try:
            from transformers.utils import logging as tl
            tl.disable_progress_bar()
        except Exception:
            pass
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except Exception:
            pass

    def _dtype(self):
        import torch
        if torch.cuda.is_available():
            return torch.bfloat16 if self.USE_BFLOAT16 else torch.float16
        return torch.float32

    def _load_one(self, model_id: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from .translatorOptim import apply_torch_compile, from_pretrained_with_attention

        tok_kwargs: dict = {"trust_remote_code": True}
        if self.SRC_LANG:
            tok_kwargs["src_lang"] = self.SRC_LANG
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)

        model = from_pretrained_with_attention(
            AutoModelForSeq2SeqLM,
            model_id,
            self.LABEL,
            quantization_config=None,
            device_map="auto",
            dtype=self._dtype(),
            low_cpu_mem_usage=True,
        )
        model.eval()
        model = apply_torch_compile(model, self.LABEL)
        return model, tokenizer

    def _load(self):
        import torch
        self._suppress_hf_logging()

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            dtype_label = "bfloat16" if self.USE_BFLOAT16 else "float16"
            print(f"\nLoading {self.LABEL}... GPU: {gpu_name}, VRAM: {total_gb:.2f}GB, dtype={dtype_label}")
        else:
            print(f"\nLoading {self.LABEL}... WARNING: No CUDA GPU detected, CPU mode.")

        model_ids = self.MODEL_IDS or [self.MODEL_ID]
        last_error: Exception | None = None
        for model_id in model_ids:
            try:
                model, tokenizer = self._load_one(model_id)
                size = f" ({self.MODEL_SIZE_MSG})" if self.MODEL_SIZE_MSG else ""
                print(f"Model loaded: {model_id}{size}")
                return model, tokenizer
            except Exception as exc:
                last_error = exc
                if len(model_ids) > 1:
                    print(f"Could not load {model_id}: {exc}")

        raise RuntimeError(f"{self.LABEL} model load failed.") from last_error

    def ensure_model_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            try:
                self._model, self._tokenizer = self._load()
            except ImportError as exc:
                raise RuntimeError(f"Model load failed: {exc}") from exc

    def _input_device(self):
        import torch
        for p in self._model.parameters():
            if str(p.device) != "meta":
                return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_text(self, text: str) -> str:
        t = text.strip()
        return f"{self.SOURCE_PREFIX}{t}" if self.SOURCE_PREFIX else t

    def _gen_kwargs(self) -> dict:
        kwargs: dict = {
            "max_new_tokens": self.MAX_LENGTH,
            "num_beams": 4,
            "early_stopping": True,
            "do_sample": False,
        }
        if self.FORCED_BOS_LANG:
            kwargs["forced_bos_token_id"] = self._tokenizer.convert_tokens_to_ids(self.FORCED_BOS_LANG)
        return kwargs

    def _set_src_lang(self) -> None:
        if self.SRC_LANG:
            self._tokenizer.src_lang = self.SRC_LANG

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        self.ensure_model_loaded()
        import torch
        self._set_src_lang()
        inputs = self._tokenizer(
            self._prepare_text(text), return_tensors="pt", max_length=self.MAX_LENGTH, truncation=True
        )
        inputs = {k: v.to(self._input_device()) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **self._gen_kwargs())
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def translate_many(self, texts: list[str], batch_size: int = 4) -> list[str]:
        if not texts:
            return []
        self.ensure_model_loaded()
        import torch
        self._set_src_lang()
        results: list[str] = []
        gen_kwargs = self._gen_kwargs()
        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start : start + max(1, batch_size)]
            prompts = [self._prepare_text(t) if t and t.strip() else "" for t in batch]
            inputs = self._tokenizer(
                prompts, return_tensors="pt", max_length=self.MAX_LENGTH, padding=True, truncation=True
            )
            inputs = {k: v.to(self._input_device()) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)
            results.extend(t.strip() for t in self._tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return results

    def count_tokens_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        self.ensure_model_loaded()
        self._set_src_lang()
        prompts = [self._prepare_text(t) if t and t.strip() else "" for t in texts]
        encoded = self._tokenizer(prompts, add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def unload_model(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
