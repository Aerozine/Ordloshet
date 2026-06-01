"""Gemma Translate 12B direct translation adapter."""
from .base_llm import LLMBase


class _Gemma(LLMBase):
    MODEL_ID = "google/translategemma-12b-it"
    LABEL = "Gemma"
    USE_BFLOAT16 = True
    MODEL_SIZE_MSG = "~6GB"
    MAX_INPUT_TOKENS = 1536

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        self.ensure_model_loaded()
        # TranslateGemma's chat template rejects system-first conversations on some versions.
        prompt = (
            "Translate the following English fiction passage to natural published French. "
            "Return only the French translation, with no notes. Use informal dialogue when the "
            "characters address each other directly. Preserve names, numbers, paragraph meaning, "
            "tone, and profanity. Do not add dates, explanations, or details that are not in the "
            "English text.\n\n"
            f"English:\n{text.strip()}\n\nFrench:"
        )
        return self._generate_from_prompt(prompt)


_inst = _Gemma()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
unload_model = _inst.unload_model
largetranslate = large_translate
