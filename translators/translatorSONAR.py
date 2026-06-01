"""Meta SONAR text-to-text translation adapter (fairseq2)."""
from __future__ import annotations

_pipeline = None


def _load_pipeline():
    try:
        from sonar.inference_pipelines.text import TextToTextModelPipeline
    except ImportError as exc:
        raise RuntimeError(
            "sonar-space is required for SONAR. Run: pip install sonar-space"
        ) from exc

    print("\nLoading SONAR text encoder/decoder...")
    pipeline = TextToTextModelPipeline(
        encoder="text_sonar_basic_encoder",
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
    )
    print("SONAR pipeline ready.")
    return pipeline


def ensure_model_loaded() -> None:
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_pipeline()


def translate(text: str) -> str:
    if not text or not text.strip():
        return text
    ensure_model_loaded()
    results = _pipeline.predict([text.strip()], source_lang="eng_Latn", target_lang="fra_Latn")
    return results[0] if results else text


def translate_many(texts: list[str], batch_size: int = 16) -> list[str]:
    if not texts:
        return []
    ensure_model_loaded()
    outputs: list[str] = []
    for start in range(0, len(texts), max(1, batch_size)):
        batch = texts[start : start + max(1, batch_size)]
        valid_indices = [i for i, t in enumerate(batch) if t and t.strip()]
        valid_texts = [batch[i].strip() for i in valid_indices]
        results: list[str] = []
        if valid_texts:
            results = list(
                _pipeline.predict(valid_texts, source_lang="eng_Latn", target_lang="fra_Latn")
            )
        vi = 0
        for i, t in enumerate(batch):
            if i in valid_indices:
                outputs.append(results[vi])
                vi += 1
            else:
                outputs.append(t)
    return outputs


def count_tokens_many(texts: list[str]) -> list[int]:
    # SONAR uses its own tokenizer; approximate with whitespace split for token budget
    return [len(t.split()) if t else 0 for t in texts]


def unload_model() -> None:
    global _pipeline
    _pipeline = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


large_translate = translate
largetranslate = translate
