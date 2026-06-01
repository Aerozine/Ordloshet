"""Mistral-7B-Instruct AWQ 4-bit adapter — same API as translatorMistral, less VRAM."""
from .base_llm_awq import LLMBaseAWQ


class _MistralAWQ(LLMBaseAWQ):
    MODEL_ID = "solidrust/Mistral-7B-Instruct-v0.3-AWQ"
    LABEL = "Mistral-AWQ"
    MODEL_SIZE_MSG = "~4GB (AWQ 4-bit)"
    TRANSLATE_SYSTEM = (
        "You are a professional French literary translator. Return only the French "
        "translation, with no notes. Preserve names, numbers, paragraph meaning, tone, "
        "and profanity."
    )


_inst = _MistralAWQ()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
build_chapter_memo = _inst.build_chapter_memo
patch_translation = _inst.patch_translation
repair_translation = _inst.repair_translation
unload_model = _inst.unload_model
largetranslate = large_translate
