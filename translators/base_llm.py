"""Base class for causal-LLM translation and post-editing adapters."""
from __future__ import annotations


class LLMBase:
    MODEL_ID: str = ""
    LABEL: str = ""
    USE_BFLOAT16: bool = False
    MODEL_SIZE_MSG: str = ""
    MAX_INPUT_TOKENS: int = 2048
    MAX_NEW_TOKENS: int = 512
    TRUST_REMOTE_CODE: bool = True
    TRANSLATE_SYSTEM: str | None = None   # system message for translate(); None = no system role

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

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .translatorOptim import apply_torch_compile, from_pretrained_with_attention

        self._suppress_hf_logging()
        dtype = self._dtype()

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            dtype_label = "bfloat16" if self.USE_BFLOAT16 else "float16"
            print(f"\nLoading {self.LABEL}... GPU: {gpu_name}, VRAM: {total_gb:.2f}GB, dtype={dtype_label}")
        else:
            print(f"\nLoading {self.LABEL}... WARNING: No CUDA GPU detected, CPU mode.")

        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = from_pretrained_with_attention(
            AutoModelForCausalLM,
            self.MODEL_ID,
            self.LABEL,
            device_map="auto",
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=self.TRUST_REMOTE_CODE,
        )
        model.eval()
        model = apply_torch_compile(model, self.LABEL)
        size = f" ({self.MODEL_SIZE_MSG})" if self.MODEL_SIZE_MSG else ""
        print(f"Model loaded{size}")
        return model, tokenizer

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

    def _chat_prompt(self, user_text: str, system_text: str | None = None) -> str:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prefix = f"{system_text}\n\n" if system_text else ""
            return f"{prefix}{user_text}"

    def _generate_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        max_input_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        import torch
        if max_new_tokens is None:
            max_new_tokens = self.MAX_NEW_TOKENS
        if max_input_tokens is None:
            max_input_tokens = self.MAX_INPUT_TOKENS
        inputs = self._tokenizer(
            prompt, return_tensors="pt", max_length=max_input_tokens, truncation=True
        )
        inputs = {k: v.to(self._input_device()) for k, v in inputs.items()}
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.9
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        self.ensure_model_loaded()
        prompt = self._chat_prompt(
            "Translate the following English literary passage to natural published French. "
            "Return only the French translation, with no notes. Use natural published French "
            "and informal dialogue when characters address each other directly. Preserve names, "
            "numbers, paragraph meaning, tone, and profanity. Do not add dates, explanations, "
            "or details that are not in the English text. Preserve placeholders like [[ZXEM1X]].\n\n"
            f"English:\n{text.strip()}\n\nFrench:",
            self.TRANSLATE_SYSTEM,
        )
        return self._generate_from_prompt(prompt)

    def build_chapter_memo(self, sample_text: str) -> str:
        self.ensure_model_loaded()
        prompt = self._chat_prompt(
            "Create a concise French translation style memo from these English source and "
            "French draft samples. Include names, register, tense, recurring terms, and "
            "dialogue style. Do not translate the chapter.\n\n"
            f"{sample_text}\n\nMemo:",
            "You are a literary translation editor.",
        )
        return self._generate_from_prompt(prompt, max_new_tokens=256)

    def patch_translation(
        self,
        source: str,
        draft: str,
        context: str,
        memo: str,
        issues: str = "",
        alt_drafts: str = "",
        correction_examples: str = "",
        register_hint: str = "",
        temperature: float = 0.0,
    ) -> str:
        import re as _re
        self.ensure_model_loaded()
        alternate_section = (
            f"ALTERNATE MACHINE DRAFTS:\n{alt_drafts}\n\n" if alt_drafts.strip() else ""
        )
        memory_section = (
            f"ACCEPTED CORRECTION MEMORY EXAMPLES:\n{correction_examples}\n\n"
            if correction_examples.strip()
            else ""
        )
        # Extract proper names from source so the model knows what to keep verbatim.
        names = _re.findall(r"\b([A-Z][a-z]{1,}(?:\s[A-Z][a-z]+)*)\b", source)
        # De-duplicate, skip common sentence-start false positives
        seen: set[str] = set()
        unique_names = []
        for n in names:
            if n not in seen and len(n) > 2:
                seen.add(n)
                unique_names.append(n)
        name_constraint = (
            f"Never omit or translate these names: {', '.join(unique_names[:8])}.\n"
            if unique_names else ""
        )
        # Remind the model of the exact marker syntax used in the draft.
        marker_ids = _re.findall(r"\[\[(ZX[A-Z0-9]+X)\]\]", draft)
        marker_constraint = (
            f"The draft contains inline markers {', '.join(dict.fromkeys(marker_ids))} — "
            "preserve each [[ID]]...[[/ID]] pair exactly as written.\n"
            if marker_ids else ""
        )
        register_constraint = ""
        if register_hint == "informal":
            register_constraint = (
                "Register constraint: when the source says you/your in direct address, "
                "use informal tu/toi/te/ton/ta/tes, not vous/votre.\n"
            )
        elif register_hint == "plural_or_formal_allowed":
            register_constraint = (
                "Register constraint: vous is allowed only for plural or explicitly formal address.\n"
            )
        src_words = len(source.split())
        length_constraint = (
            f"Length constraint: the French output must be {max(1, src_words - 4)}–"
            f"{src_words + 8} words (source is {src_words} words). "
            "Do not omit sentences or shorten clauses that are present in the source.\n"
        )
        prompt = self._chat_prompt(
            "Correct the French draft using the English source.\n"
            "Return only the corrected French target chunk, with no notes.\n"
            "Do not add, merge, or omit sentences relative to the source.\n"
            f"{length_constraint}"
            f"{name_constraint}"
            f"{marker_constraint}"
            f"{register_constraint}"
            "Follow the entity graph and style rules from the memo: speaker, addressee, "
            "scene participants, register, aliases, and tense.\n"
            "Preserve dialogue punctuation, negation, tense, and profanity strength.\n"
            "Only rewrite to fix the listed issues.\n\n"
            f"STYLE MEMO:\n{memo}\n\n"
            f"{memory_section}"
            f"PATCH REASONS:\n{issues or '- selected for review'}\n\n"
            f"LOCAL CONTEXT:\n{context}\n\n"
            f"SOURCE TARGET CHUNK:\n{source}\n\n"
            f"FRENCH DRAFT TARGET CHUNK:\n{draft}\n\n"
            f"{alternate_section}"
            "Corrected French:",
            "You are a careful French literary post-editor.",
        )
        return self._generate_from_prompt(
            prompt,
            max_input_tokens=4096,
            temperature=temperature,
        )

    def repair_translation(
        self,
        source: str,
        draft: str,
        bad_translation: str,
        flags: list[str],
        context: str,
        memo: str,
    ) -> str:
        self.ensure_model_loaded()
        prompt = self._chat_prompt(
            "Repair this rejected post-edit. Return only one corrected French chunk.\n"
            "Preserve placeholders exactly and stay close to the source meaning.\n"
            "Follow the entity graph and style rules from the memo.\n"
            "Preserve dialogue punctuation, negation, names, tense, and profanity strength.\n"
            f"Rejected flags: {', '.join(flags)}\n\n"
            f"STYLE MEMO:\n{memo}\n\n"
            f"SOURCE:\n{source}\n\n"
            f"DRAFT:\n{draft}\n\n"
            f"BAD OUTPUT:\n{bad_translation}\n\n"
            "Repaired French:",
            "You are a careful French literary post-editor.",
        )
        return self._generate_from_prompt(prompt, max_input_tokens=4096)

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
