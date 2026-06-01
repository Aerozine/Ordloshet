"""Mistral-7B-Instruct LLM post-editor adapter."""
from .base_llm import LLMBase


class _Mistral(LLMBase):
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    LABEL = "Mistral"
    MODEL_SIZE_MSG = "~4.5GB"
    TRANSLATE_SYSTEM = (
        "You are a professional French literary translator. Return only the French "
        "translation, with no notes. Use natural published French and informal dialogue "
        "when characters address each other directly. Preserve names, numbers, paragraph "
        "meaning, tone, and profanity. Do not add dates, explanations, or details that "
        "are not in the English text."
    )


_inst = _Mistral()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
build_chapter_memo = _inst.build_chapter_memo
patch_translation = _inst.patch_translation
repair_translation = _inst.repair_translation
unload_model = _inst.unload_model
largetranslate = large_translate
