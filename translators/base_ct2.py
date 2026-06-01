"""Base class for CTranslate2 MT adapters (NLLB CT2, OPUS CT2, MADLAD CT2)."""
from __future__ import annotations

from pathlib import Path


class CT2TranslatorBase:
    MODEL_ID: str = ""
    CT2_DIR: Path = Path("models/ct2/model")
    LABEL: str = ""
    SOURCE_PREFIX: str = ""  # MADLAD: "<2fr> "; NLLB/OPUS: ""
    SRC_LANG: str = ""       # NLLB: "eng_Latn"; OPUS/MADLAD: ""
    TGT_LANG: str = ""       # NLLB: "fra_Latn"; OPUS/MADLAD: ""

    def __init__(self) -> None:
        self._translator = None
        self._tokenizer = None

    def _load(self):
        import ctranslate2
        from transformers import AutoTokenizer
        from .translatorCT2Common import ct2_compute_type, ct2_device, ensure_ct2_model

        model_dir = ensure_ct2_model(self.MODEL_ID, self.CT2_DIR)
        device = ct2_device()
        compute_type = ct2_compute_type()
        print(f"\nLoading {self.LABEL} CTranslate2 on {device}, compute_type={compute_type}...")
        translator = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type)

        tok_kwargs: dict = {"trust_remote_code": True}
        if self.SRC_LANG:
            tok_kwargs["src_lang"] = self.SRC_LANG
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, **tok_kwargs)
        print(f"CTranslate2 {self.LABEL} model ready.")
        return translator, tokenizer

    def ensure_model_loaded(self) -> None:
        if self._translator is None or self._tokenizer is None:
            try:
                self._translator, self._tokenizer = self._load()
            except ImportError as exc:
                raise RuntimeError(f"CTranslate2 {self.LABEL} load failed: {exc}") from exc

    def _encode(self, text: str) -> list[str]:
        if self.SRC_LANG:
            self._tokenizer.src_lang = self.SRC_LANG
        source = f"{self.SOURCE_PREFIX}{text.strip()}" if self.SOURCE_PREFIX else text.strip()
        return self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(source))

    def _decode(self, tokens: list[str]) -> str:
        if self.TGT_LANG and tokens and tokens[0] == self.TGT_LANG:
            tokens = tokens[1:]
        ids = self._tokenizer.convert_tokens_to_ids(tokens)
        return self._tokenizer.decode(ids, skip_special_tokens=True).strip()

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        return self.translate_many([text], 1)[0]

    def _base_kwargs(self, batch_size: int, beam_size: int = 4, n_hyp: int = 1) -> dict:
        return {
            "max_batch_size": max(1, batch_size),
            "batch_type": "examples",
            "beam_size": beam_size,
            "num_hypotheses": n_hyp,
            "max_decoding_length": 512,
            "repetition_penalty": 1.2,
        }

    def translate_many(self, texts: list[str], batch_size: int = 4) -> list[str]:
        if not texts:
            return []
        self.ensure_model_loaded()
        outputs: list[str] = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start : start + max(1, batch_size)]
            sources = [self._encode(t) if t and t.strip() else [] for t in batch]
            kwargs = self._base_kwargs(batch_size)
            if self.TGT_LANG:
                kwargs["target_prefix"] = [[self.TGT_LANG] for _ in sources]
            results = self._translator.translate_batch(sources, **kwargs)
            for original, result in zip(batch, results):
                outputs.append(original if not original or not original.strip()
                                else self._decode(result.hypotheses[0]))
        return outputs

    def translate_many_nbest(self, texts: list[str], batch_size: int = 4, n: int = 4) -> list[list[str]]:
        """Return up to n translation hypotheses per input, scored by beam rank."""
        if not texts:
            return []
        self.ensure_model_loaded()
        all_hyps: list[list[str]] = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start : start + max(1, batch_size)]
            sources = [self._encode(t) if t and t.strip() else [] for t in batch]
            kwargs = self._base_kwargs(batch_size, beam_size=max(n, 4), n_hyp=n)
            if self.TGT_LANG:
                kwargs["target_prefix"] = [[self.TGT_LANG] for _ in sources]
            results = self._translator.translate_batch(sources, **kwargs)
            for original, result in zip(batch, results):
                if not original or not original.strip():
                    all_hyps.append([original])
                else:
                    hyps = [self._decode(h) for h in result.hypotheses if h]
                    all_hyps.append(hyps or [original])
        return all_hyps

    def count_tokens_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        self.ensure_model_loaded()
        if self.SRC_LANG:
            self._tokenizer.src_lang = self.SRC_LANG
        encoded = self._tokenizer(texts, add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def unload_model(self) -> None:
        if self._translator is not None:
            try:
                self._translator.unload_model()
            except Exception:
                pass
        self._translator = None
        self._tokenizer = None
        from .translatorCT2Common import cleanup_cuda
        cleanup_cuda()
