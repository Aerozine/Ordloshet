"""Qwen 2.5-7B-Instruct AWQ 4-bit adapter — same API as translatorQwen, less VRAM."""
import json

from .base_llm_awq import LLMBaseAWQ


class _QwenAWQ(LLMBaseAWQ):
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    LABEL = "Qwen-AWQ"
    MODEL_SIZE_MSG = "~4GB (AWQ 4-bit)"
    MAX_INPUT_TOKENS = 3072
    TRANSLATE_SYSTEM = "You are a professional French literary translator."

    def critique_translation(self, source, draft, candidate, context, memo):
        self.ensure_model_loaded()
        prompt = self._chat_prompt(
            "Evaluate whether the French candidate preserves the English source meaning.\n"
            "Return exactly OK if faithful and grammatical. Otherwise return short bullet issues only.\n"
            "Check omissions, additions, negation, tense, register, profanity, and placeholders.\n\n"
            f"STYLE MEMO:\n{memo}\n\n"
            f"LOCAL CONTEXT:\n{context}\n\n"
            f"SOURCE:\n{source}\n\nDRAFT:\n{draft}\n\nCANDIDATE:\n{candidate}\n\nCritique:",
            "You are a strict bilingual QA reviewer for literary translation.",
        )
        return self._generate_from_prompt(prompt, max_new_tokens=192, max_input_tokens=4096)

    def reread_translation(self, source, current, context, memo, issues=""):
        self.ensure_model_loaded()
        prompt = self._chat_prompt(
            "Improve only the target chunk in this French chapter window.\n"
            "Return only the corrected French chunk, no notes.\n"
            "Preserve source meaning, placeholders, and dialogue punctuation. Follow memo style.\n\n"
            f"STYLE MEMO:\n{memo}\n\nISSUES:\n{issues or '- chapter reread'}\n\n"
            f"CONTEXT:\n{context}\n\nSOURCE:\n{source}\n\nCURRENT FRENCH:\n{current}\n\nReread French:",
            "You are a senior French literary translation editor.",
        )
        return self._generate_from_prompt(prompt, max_input_tokens=4096)

    def back_translate_to_english(self, candidate):
        self.ensure_model_loaded()
        prompt = self._chat_prompt(
            "Back-translate this French chunk to literal English for QA.\n"
            "Return only the English meaning, preserving names and negation.\n\n"
            f"French:\n{candidate}\n\nEnglish:",
            "You are a literal bilingual QA assistant.",
        )
        return self._generate_from_prompt(prompt, max_input_tokens=2048)

    def revise_translation_window(self, source_chunks, current_chunks, context, memo, issues=""):
        self.ensure_model_loaded()
        payload = {"source_chunks": source_chunks, "current_french_chunks": current_chunks}
        prompt = self._chat_prompt(
            "Revise this window of French translation chunks.\n"
            "Return strict JSON: array of {\"index\": number, \"translation\": string}.\n"
            "Same count, order, indices as input. Preserve placeholders, names, and dialogue.\n\n"
            f"STYLE MEMO:\n{memo}\n\nISSUES:\n{issues or '- paragraph-window reread'}\n\n"
            f"CONTEXT:\n{context}\n\nWINDOW JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\nJSON:",
            "You are a senior French literary translation editor returning JSON.",
        )
        return self._generate_from_prompt(prompt, max_new_tokens=1024, max_input_tokens=6144)


_inst = _QwenAWQ()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
build_chapter_memo = _inst.build_chapter_memo
patch_translation = _inst.patch_translation
repair_translation = _inst.repair_translation
critique_translation = _inst.critique_translation
reread_translation = _inst.reread_translation
back_translate_to_english = _inst.back_translate_to_english
revise_translation_window = _inst.revise_translation_window
unload_model = _inst.unload_model
largetranslate = large_translate
