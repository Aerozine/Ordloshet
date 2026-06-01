"""Cross-chapter style memo persistence.

Loads and saves a book-level cumulative style memo that grows across chapters.
Stored in: cache/book_memory/{book_slug}_style_memo.txt
"""
from __future__ import annotations

import re
from pathlib import Path

DEFAULT_MEMO_DIR = Path("cache") / "book_memory"


def _slug(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9_-]", "_", text.lower()).strip("_")[:80]


def load_persistent_memo(book_path: str | Path, memo_dir: str | Path = DEFAULT_MEMO_DIR) -> str:
    """Return the persisted cross-chapter style memo, or empty string if none."""
    path = Path(memo_dir) / f"{_slug(Path(book_path).stem)}_style_memo.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def save_persistent_memo(
    book_path: str | Path,
    new_chapter_memo: str,
    patcher_module,
    memo_dir: str | Path = DEFAULT_MEMO_DIR,
) -> str:
    """Merge this chapter's style memo into the persisted memo and save it.

    Returns the updated memo text.
    """
    path = Path(memo_dir) / f"{_slug(Path(book_path).stem)}_style_memo.txt"
    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8").strip()

    if not existing:
        updated = new_chapter_memo
    else:
        updated = _merge_memos(existing, new_chapter_memo, patcher_module)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return updated


def _merge_memos(existing: str, new_chapter: str, patcher_module) -> str:
    """Ask the LLM to merge two style memos, or concatenate if LLM unavailable."""
    chat_fn = getattr(patcher_module, "_chat_prompt", None)
    gen_fn = getattr(patcher_module, "_generate_from_prompt", None)
    ensure = getattr(patcher_module, "ensure_model_loaded", None)

    if not callable(chat_fn) or not callable(gen_fn):
        # No LLM available — simple concatenation keeping latest chapter's new items
        return f"{existing}\n\n--- Updated from new chapter ---\n{new_chapter}"

    if callable(ensure):
        ensure()

    prompt = chat_fn(
        "Merge these two French translation style memos into one concise updated memo.\n"
        "Keep all names, register rules, tense rules, and recurring term choices from both.\n"
        "Remove redundant lines. Prefer the newer chapter's phrasing when there is a conflict.\n"
        "Return only the merged memo, no additional commentary.\n\n"
        f"EXISTING MEMO:\n{existing}\n\n"
        f"NEW CHAPTER MEMO:\n{new_chapter}\n\n"
        "Merged memo:",
        "You are a literary translation style editor.",
    )
    try:
        return gen_fn(prompt, max_new_tokens=512, max_input_tokens=3000)
    except Exception:
        return f"{existing}\n\n--- Updated from new chapter ---\n{new_chapter}"
