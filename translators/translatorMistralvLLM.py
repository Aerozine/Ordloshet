"""Mistral-7B-Instruct vLLM adapter — high-throughput batch patcher."""
from __future__ import annotations

_llm = None
_tokenizer = None
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
_TRANSLATE_SYSTEM = (
    "You are a professional French literary translator. Return only the French "
    "translation, with no notes. Use natural published French and informal dialogue "
    "when characters address each other directly. Preserve names, numbers, paragraph "
    "meaning, tone, and profanity."
)


def _load():
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "vllm is required for this adapter. Run: pip install vllm"
        ) from exc

    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"\nLoading {MODEL_ID} via vLLM...")
    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=0.6,
        dtype="float16",
        enable_prefix_caching=True,   # reuse KV cache for shared prompt prefix (style memo)
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("vLLM Mistral ready.")
    return llm, tokenizer


def ensure_model_loaded() -> None:
    global _llm, _tokenizer
    if _llm is None or _tokenizer is None:
        _llm, _tokenizer = _load()


def _chat_prompt(user_text: str, system_text: str | None = None) -> str:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    try:
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prefix = f"{system_text}\n\n" if system_text else ""
        return f"{prefix}{user_text}"


def _generate_batch(prompts: list[str], max_new_tokens: int = 512, temperature: float = 0.0) -> list[str]:
    from vllm import SamplingParams
    params = SamplingParams(temperature=temperature, top_p=0.9 if temperature > 0 else 1.0, max_tokens=max_new_tokens)
    outputs = _llm.generate(prompts, params)
    return [o.outputs[0].text.strip() for o in outputs]


def translate(text: str) -> str:
    if not text or not text.strip():
        return text
    ensure_model_loaded()
    prompt = _chat_prompt(
        "Translate the following English literary passage to natural published French. "
        "Return only the French translation. Preserve placeholders like [[ZXEM1X]].\n\n"
        f"English:\n{text.strip()}\n\nFrench:",
        _TRANSLATE_SYSTEM,
    )
    return _generate_batch([prompt])[0]


def build_chapter_memo(sample_text: str) -> str:
    ensure_model_loaded()
    prompt = _chat_prompt(
        "Create a concise French translation style memo from these English source and "
        "French draft samples. Include names, register, tense, recurring terms, and "
        "dialogue style. Do not translate.\n\n"
        f"{sample_text}\n\nMemo:"
    )
    return _generate_batch([prompt], max_new_tokens=256)[0]


def patch_translation(
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
    ensure_model_loaded()
    register_section = ""
    if register_hint == "informal":
        register_section = "Use informal tu/toi/te/ton/ta/tes for direct you/your address; avoid vous/votre.\n"
    elif register_hint == "plural_or_formal_allowed":
        register_section = "Vous is allowed only for plural or explicitly formal address.\n"
    prompt = _chat_prompt(
        "Correct the French draft using the English source.\n"
        "Return only the corrected French chunk, with no notes.\n"
        "Preserve all placeholders like [[ZXEM1X]], names, dialogue punctuation, "
        "negation, tense, and profanity. Follow memo style rules.\n"
        f"{register_section}"
        "Only rewrite to fix the listed issues.\n\n"
        f"STYLE MEMO:\n{memo}\n\n"
        f"PATCH REASONS:\n{issues or '- selected for review'}\n\n"
        f"LOCAL CONTEXT:\n{context}\n\n"
        f"SOURCE:\n{source}\n\n"
        f"DRAFT:\n{draft}\n\n"
        "Corrected French:",
        "You are a careful French literary post-editor.",
    )
    return _generate_batch([prompt], temperature=temperature)[0]


def batch_patch_translations(items: list[dict]) -> list[str]:
    """True batch patching for throughput — items: list of patch_translation kwargs dicts."""
    ensure_model_loaded()
    prompts = [
        _chat_prompt(
            "Correct the French draft using the English source.\n"
            "Return only the corrected French chunk, with no notes.\n"
            "Preserve all placeholders, names, dialogue punctuation. Follow memo style.\n"
            f"{'Use informal tu/toi/te/ton/ta/tes for direct you/your address; avoid vous/votre. ' if item.get('register_hint') == 'informal' else ''}"
            "Only rewrite to fix the listed issues.\n\n"
            f"STYLE MEMO:\n{item.get('memo', '')}\n\n"
            f"PATCH REASONS:\n{item.get('issues', '') or '- selected for review'}\n\n"
            f"LOCAL CONTEXT:\n{item.get('context', '')}\n\n"
            f"SOURCE:\n{item.get('source', '')}\n\n"
            f"DRAFT:\n{item.get('draft', '')}\n\n"
            "Corrected French:",
            "You are a careful French literary post-editor.",
        )
        for item in items
    ]
    return _generate_batch(prompts)


def repair_translation(
    source: str,
    draft: str,
    bad_translation: str,
    flags: list[str],
    context: str,
    memo: str,
) -> str:
    ensure_model_loaded()
    prompt = _chat_prompt(
        "Repair this rejected post-edit. Return only one corrected French chunk.\n"
        "Preserve placeholders exactly. Follow memo style rules.\n"
        f"Rejected flags: {', '.join(flags)}\n\n"
        f"STYLE MEMO:\n{memo}\n\n"
        f"SOURCE:\n{source}\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"BAD OUTPUT:\n{bad_translation}\n\n"
        "Repaired French:",
        "You are a careful French literary post-editor.",
    )
    return _generate_batch([prompt])[0]


def unload_model() -> None:
    global _llm, _tokenizer
    _llm = None
    _tokenizer = None
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
