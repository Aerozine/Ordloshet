"""Automatic glossary extraction from source chapter text using an LLM.

Usage:
    from lib.glossary_extractor import extract_glossary
    rules = extract_glossary(source_text, patcher_module)
    # rules is a dict compatible with GLOSSARY_RULES format
"""
from __future__ import annotations

import json
import re


_EXTRACTION_SYSTEM = (
    "You are a terminology extractor for English-to-French literary translation. "
    "Return only a JSON array, no prose."
)

_EXTRACTION_PROMPT = """\
Analyse the following English fiction passage and identify terms that need consistent \
French translation: sport-specific vocabulary, recurring proper objects, idiomatic phrases, \
and culturally-loaded nouns. Exclude character names and common words.

For each term return a JSON object with:
  "term": the English term (lowercase)
  "preferred_fr": list of acceptable French translations (regex patterns)
  "forbidden_fr": list of French translations to avoid (regex patterns, may be empty)
  "source_pattern": regex matching the English term (word-boundary aware)

Return a JSON array of these objects. Return [] if nothing notable found.

TEXT:
{text}

JSON:"""


def extract_glossary(source_text: str, patcher_module) -> dict[str, dict]:
    """Run the LLM to extract glossary rules from source_text.

    Returns a dict in the GLOSSARY_RULES format:
      { "term_label": {"source": [...], "preferred": [...], "forbidden": [...]} }
    """
    ensure = getattr(patcher_module, "ensure_model_loaded", None)
    if callable(ensure):
        ensure()

    chat_prompt_fn = getattr(patcher_module, "_chat_prompt", None)
    generate_fn = getattr(patcher_module, "_generate_from_prompt", None)

    if not callable(chat_prompt_fn) or not callable(generate_fn):
        # Fall back to module-level translation API (non-LLM models can't extract)
        return {}

    preview = source_text[:4000].strip()
    prompt = chat_prompt_fn(
        _EXTRACTION_PROMPT.format(text=preview),
        _EXTRACTION_SYSTEM,
    )

    try:
        raw = generate_fn(prompt, max_new_tokens=512, max_input_tokens=5000)
    except Exception:
        return {}

    return _parse_extraction(raw)


def _parse_extraction(raw: str) -> dict[str, dict]:
    raw = raw.strip()
    # Try to find a JSON array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return {}
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return {}

    rules: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        source_pat = item.get("source_pattern") or [rf"\b{re.escape(term)}\b"]
        if isinstance(source_pat, str):
            source_pat = [source_pat]
        preferred = item.get("preferred_fr", [])
        if isinstance(preferred, str):
            preferred = [preferred]
        forbidden = item.get("forbidden_fr", [])
        if isinstance(forbidden, str):
            forbidden = [forbidden]

        label = term.replace(" ", "_")
        rules[label] = {
            "source": [str(p) for p in source_pat],
            "preferred": [str(p) for p in preferred],
            "forbidden": [str(p) for p in forbidden],
        }
    return rules


def merge_into_glossary(existing: dict, extracted: dict) -> dict:
    """Merge extracted rules into existing GLOSSARY_RULES without overwriting manual entries."""
    merged = dict(existing)
    for key, rule in extracted.items():
        if key not in merged:
            merged[key] = rule
    return merged
