"""Qwen 2.5-7B-Instruct LLM post-editor adapter (full patcher capabilities)."""
import json

from .base_llm import LLMBase


class _Qwen(LLMBase):
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    LABEL = "Qwen"
    MODEL_SIZE_MSG = "~7B"
    MAX_INPUT_TOKENS = 3072
    TRANSLATE_SYSTEM = "You are a professional French literary translator."

    def critique_translation(
        self,
        source: str,
        draft: str,
        candidate: str,
        context: str,
        memo: str,
    ) -> str:
        self.ensure_model_loaded()
        prompt = self._chat_prompt(
            "Evaluate whether the French candidate preserves the English source meaning.\n"
            "Return exactly OK if it is faithful and grammatical.\n"
            "Otherwise return short bullet issues only; do not rewrite.\n"
            "Check omissions, additions, wrong subject/object, negation, tense, register, "
            "profanity, and lost inline placeholders.\n"
            "Use the entity graph and style rules from the memo when judging speaker/addressee "
            "and tu/vous.\n\n"
            f"STYLE MEMO:\n{memo}\n\n"
            f"LOCAL CONTEXT:\n{context}\n\n"
            f"SOURCE TARGET CHUNK:\n{source}\n\n"
            f"FRENCH DRAFT TARGET CHUNK:\n{draft}\n\n"
            f"FRENCH CANDIDATE TARGET CHUNK:\n{candidate}\n\n"
            "Critique:",
            "You are a strict bilingual QA reviewer for literary translation.",
        )
        return self._generate_from_prompt(prompt, max_new_tokens=192, max_input_tokens=4096)

    def reread_translation(
        self,
        source: str,
        current: str,
        context: str,
        memo: str,
        issues: str = "",
    ) -> str:
        self.ensure_model_loaded()
        prompt = self._chat_prompt(
            "Reread this French chapter window and improve only the target chunk.\n"
            "Return only the corrected French target chunk, with no notes.\n"
            "Preserve the source meaning, paragraph scope, names, inline placeholders, "
            "and dialogue punctuation.\n"
            "Follow the entity graph and style rules from the memo.\n"
            "Do not add facts that are not in the source.\n\n"
            f"STYLE MEMO:\n{memo}\n\n"
            f"ISSUES TO CHECK:\n{issues or '- chapter reread'}\n\n"
            f"LOCAL CHAPTER WINDOW:\n{context}\n\n"
            f"SOURCE TARGET CHUNK:\n{source}\n\n"
            f"CURRENT FRENCH TARGET CHUNK:\n{current}\n\n"
            "Reread French:",
            "You are a senior French literary translation editor.",
        )
        return self._generate_from_prompt(prompt, max_input_tokens=4096)

    def back_translate_to_english(self, candidate: str) -> str:
        self.ensure_model_loaded()
        prompt = self._chat_prompt(
            "Back-translate this French target chunk to literal English for quality checking.\n"
            "Return only the English meaning, with names and negation preserved.\n\n"
            f"French:\n{candidate}\n\n"
            "English:",
            "You are a literal bilingual translation QA assistant.",
        )
        return self._generate_from_prompt(prompt, max_input_tokens=2048)

    def revise_translation_window(
        self,
        source_chunks: list[dict[str, str]],
        current_chunks: list[dict[str, str]],
        context: str,
        memo: str,
        issues: str = "",
    ) -> str:
        self.ensure_model_loaded()
        payload = {"source_chunks": source_chunks, "current_french_chunks": current_chunks}
        prompt = self._chat_prompt(
            "Revise a small window of French literary translation chunks.\n"
            "Return strict JSON only: an array with the same number of objects as input, "
            "each object exactly {\"index\": number, \"translation\": string}.\n"
            "Do not merge, split, remove, or reorder chunks.\n"
            "Preserve all placeholders like [[ZXEM1X]], names, aliases, dialogue punctuation, "
            "questions, negation, tense, and profanity strength.\n"
            "Follow the entity graph and style rules from the memo.\n"
            "Only improve grammar, continuity, pronouns, register, and flow where the source "
            "supports it.\n\n"
            f"STYLE MEMO:\n{memo}\n\n"
            f"ISSUES TO CHECK:\n{issues or '- paragraph-window reread'}\n\n"
            f"LOCAL CONTEXT:\n{context}\n\n"
            f"WINDOW JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            "JSON:",
            "You are a senior French literary translation editor returning machine-parseable JSON.",
        )
        return self._generate_from_prompt(prompt, max_new_tokens=1024, max_input_tokens=6144)


_inst = _Qwen()
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
