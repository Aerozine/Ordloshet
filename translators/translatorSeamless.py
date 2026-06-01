"""Meta SeamlessM4T v2 text-to-text adapter."""
from .base_seq2seq import Seq2SeqBase


class _Seamless(Seq2SeqBase):
    MODEL_ID = "facebook/seamless-m4t-v2-large"
    LABEL = "Seamless"
    SRC_LANG = "eng"
    MODEL_SIZE_MSG = "~6GB"

    def _load_one(self, model_id: str):
        from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
        from .translatorOptim import apply_torch_compile

        processor = AutoProcessor.from_pretrained(model_id)
        model = SeamlessM4Tv2ForTextToText.from_pretrained(
            model_id,
            torch_dtype=self._dtype(),
            device_map="auto",
        )
        model.eval()
        model = apply_torch_compile(model, self.LABEL)
        return model, processor

    def _set_src_lang(self) -> None:
        pass  # processor has no src_lang attribute

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        self.ensure_model_loaded()
        import torch
        inputs = self._tokenizer(
            text=text.strip(), src_lang=self.SRC_LANG, return_tensors="pt"
        )
        inputs = {k: v.to(self._input_device()) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._model.generate(**inputs, tgt_lang="fra", generate_speech=False)
        seqs = output[0] if isinstance(output, tuple) else output
        if seqs.ndim > 1:
            seqs = seqs[0]
        return self._tokenizer.decode(seqs, skip_special_tokens=True).strip()

    def translate_many(self, texts: list[str], batch_size: int = 4) -> list[str]:
        if not texts:
            return []
        self.ensure_model_loaded()
        import torch
        results: list[str] = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start : start + max(1, batch_size)]
            valid = [t for t in batch if t and t.strip()]
            if not valid:
                results.extend(batch)
                continue
            inputs = self._tokenizer(
                text=valid, src_lang=self.SRC_LANG, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._input_device()) for k, v in inputs.items()}
            with torch.no_grad():
                output = self._model.generate(**inputs, tgt_lang="fra", generate_speech=False)
            seqs = output[0] if isinstance(output, tuple) else output
            decoded = [
                self._tokenizer.decode(row, skip_special_tokens=True).strip()
                for row in seqs
            ]
            vi = 0
            for t in batch:
                if t and t.strip():
                    results.append(decoded[vi])
                    vi += 1
                else:
                    results.append(t)
        return results

    def count_tokens_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        self.ensure_model_loaded()
        encoded = self._tokenizer(
            text=[t.strip() for t in texts],
            src_lang=self.SRC_LANG,
            add_special_tokens=True,
            truncation=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]


_inst = _Seamless()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
translate_many = _inst.translate_many
count_tokens_many = _inst.count_tokens_many
unload_model = _inst.unload_model
largetranslate = large_translate
