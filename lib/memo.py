"""Style memo building and chapter analysis utilities."""
from __future__ import annotations
import re
from lib.models import ChunkRecord, ChapterEntityGraph
from lib.char_graph import (
    strip_inline_markers, CHARACTER_ALIASES, CHARACTER_RELATIONSHIPS,
    intimate_pairs,
)
from lib.constants import TRANSLATION_STOPWORDS, KNOWN_TRANSLATION_TERMS
from lib.logging_utils import write_progress

def build_context(records: list[ChunkRecord], index: int, window: int) -> str:
    """Build patcher context window.

    The TARGET chunk gets full source+draft+meta.  Adjacent context chunks only
    send their final French text — saving ~60% of context tokens per call.
    """
    parts = []
    start = max(0, index - window)
    end = min(len(records), index + window + 1)
    for item_index in range(start, end):
        record = records[item_index]
        is_target = item_index == index
        role = "TARGET" if is_target else f"CTX {record.index}"
        draft = record.draft or ""
        final = record.final or draft

        if not is_target:
            # Context neighbours: only French output for narrative continuity
            parts.append(f"{role} FRENCH:\n{final}")
            continue

        meta_parts = []
        if record.speaker:
            meta_parts.append(f"speaker={record.speaker}")
        if record.addressee:
            meta_parts.append(f"addressee={record.addressee}")
        if record.register_hint:
            meta_parts.append(f"register={record.register_hint}")
        if record.scene_participants:
            meta_parts.append("scene=" + "/".join(record.scene_participants))
        if record.style_hints:
            meta_parts.append("rules=" + "/".join(record.style_hints))
        meta = ", ".join(meta_parts) or "speaker=unknown"
        alt_drafts = ""
        if record.alt_drafts:
            alt_lines = [
                f"{model_name}: {text}"
                for model_name, text in sorted(record.alt_drafts.items())
                if text
            ]
            if alt_lines:
                alt_drafts = "\n" + f"TARGET ALTERNATE DRAFTS:\n" + "\n".join(alt_lines)
        parts.append(
            f"TARGET META: {meta}\n"
            f"TARGET SOURCE:\n{record.source}\n"
            f"TARGET DRAFT:\n{draft}"
            f"{alt_drafts}"
        )
    return "\n\n".join(parts)


def extract_name_candidates(records: list[ChunkRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
    for record in records:
        for match in pattern.findall(strip_inline_markers(record.source)):
            if match in NAME_STOPWORDS:
                continue
            counts[match] = counts.get(match, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def extract_known_terms(records: list[ChunkRecord]) -> dict[str, str]:
    chapter_text = " ".join(strip_inline_markers(record.source).lower() for record in records)
    terms = {}
    for source_term, french_term in KNOWN_TRANSLATION_TERMS.items():
        if source_term.lower() in chapter_text:
            terms[source_term] = french_term
    return terms


def build_chapter_memory(
    records: list[ChunkRecord],
    chapter_name: str,
    args: argparse.Namespace | None = None,
) -> str:
    graph = build_chapter_entity_graph(records, chapter_name)
    names = extract_name_candidates(records)
    repeated_names = [name for name, count in names.items() if count >= 2][:20]
    known_terms = extract_known_terms(records)
    dialogue_hints = []
    for record in records:
        if record.speaker or record.addressee or record.register_hint:
            dialogue_hints.append(
                f"{record.index}: speaker={record.speaker or 'unknown'}, "
                f"addressee={record.addressee or 'unknown'}, "
                f"register={record.register_hint or 'default'}"
            )
        if len(dialogue_hints) >= 24:
            break

    lines = [
        f"Chapter: {chapter_name}",
        _build_style_line(),
        _build_register_memo_line(),
        "Dialogue punctuation: preserve spoken dialogue as dialogue; prefer French guillemets or dialogue dashes.",
        "Tense: keep past-tense narration natural in French; do not drift into present unless the source is present.",
        "Syntax: preserve every inline placeholder exactly, for example [[ZXEM1X]].",
        "Fidelity: keep paragraph scope, names, profanity intensity, jokes, and point of view.",
    ]
    lines.append(chapter_entity_graph_summary(graph))
    if repeated_names:
        lines.append("Names to keep unchanged: " + ", ".join(repeated_names))
    if known_terms:
        lines.append("Glossary:")
        for source_term, french_term in known_terms.items():
            lines.append(f"- {source_term} -> {french_term}")
    if dialogue_hints:
        lines.append("Dialogue state:")
        lines.extend(f"- {hint}" for hint in dialogue_hints)
    if args is not None:
        memory = book_memory_summary(args)
        if memory:
            lines.append(memory)
    return "\n".join(lines)


def _build_style_line() -> str:
    """Build a style description line from the book's semantic profile if available."""
    try:
        from lib.semantics import chunk_semantic_hints
        # Check if there's a stored style note from auto-analysis
        import sys
        epub_mod = sys.modules.get("epub") or sys.modules.get("__main__")
        notes = getattr(epub_mod, "_auto_style_notes", "") if epub_mod else ""
        if notes:
            return f"Style: {notes}"
    except Exception:
        pass
    return "Style: literary fiction in natural French."


def _build_register_memo_line() -> str:
    """Generate a register instruction from live CHARACTER_RELATIONSHIPS data."""
    from lib.char_graph import intimate_pairs
    pairs = intimate_pairs()
    if not pairs:
        return "Preserve all character addresses and names."
    parts = []
    for pair in pairs:
        names = sorted(pair)
        parts.append(
            f"{' and '.join(names)} are intimate partners; "
            f"use tu/toi/te/ton/ta/tes between them, never vous/votre/vos."
        )
    return " ".join(parts)


_DEFAULT_STYLE_MEMO = (
    "Literary fiction. Keep meaning, names, profanity, dialogue, and paragraph length. "
    "Use informal dialogue when characters address each other directly. "
    "Preserve inline placeholders exactly."
)


def default_style_memo() -> str:
    return _DEFAULT_STYLE_MEMO


def _trim_memo(memo: str, token_budget: int) -> str:
    """Truncate memo to approximately token_budget whitespace-split tokens."""
    if token_budget <= 0:
        return memo
    words = memo.split()
    if len(words) <= token_budget:
        return memo
    return " ".join(words[:token_budget]) + "\n[…memo trimmed]"


def build_style_memo(
    patcher_module: object,
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
    chapter_memory: str,
) -> str:
    # Inject cross-chapter persistent memo if enabled
    persistent_memo = ""
    use_persistent = (
        getattr(args, "persistent_style_memo", False)
        and getattr(args, "book_memory", False)
        and not args.dry_run
    )
    if use_persistent:
        book_stem = getattr(args, "_current_book_stem", "")
        if book_stem:
            try:
                from lib.chapter_memo import load_persistent_memo
                persistent_memo = load_persistent_memo(book_stem)
            except Exception:
                pass

    if args.dry_run or not args.llm_chapter_memo:
        return chapter_memory or default_style_memo()

    sample = []
    for record in records[:max(1, args.memo_sample_chunks)]:
        sample.append(f"SOURCE:\n{record.source}\nDRAFT:\n{record.draft}")
    sample_text = "\n\n---\n\n".join(sample)
    memo_fn = getattr(patcher_module, "build_chapter_memo", None)
    if not callable(memo_fn):
        return chapter_memory or default_style_memo()

    try:
        memo_input_parts = [f"DETERMINISTIC MEMORY:\n{chapter_memory}"]
        if persistent_memo:
            memo_input_parts.append(f"CROSS-CHAPTER STYLE MEMO:\n{persistent_memo}")
        memo_input_parts.append(f"SAMPLES:\n{sample_text}")
        memo_input = "\n\n".join(memo_input_parts)
        memo_payload = {
            "patcher": getattr(patcher_module, "__name__", "patcher"),
            "chapter": chapter_name,
            "memo_input": memo_input,
            "kind": "chapter_memo",
            "persistent_memo": bool(persistent_memo),
        }
        memo = cached_text_call(
            args,
            "chapter_memo",
            memo_payload,
            lambda: memo_fn(memo_input),
        )
        memo = memo.strip()
        if memo and use_persistent and book_stem:
            try:
                from lib.chapter_memo import save_persistent_memo
                save_persistent_memo(book_stem, memo, patcher_module)
            except Exception:
                pass
        if memo:
            return f"{chapter_memory}\n\nLLM MEMO:\n{memo}"
        return chapter_memory or default_style_memo()
    except Exception as exc:
        log_error(
            args,
            "Patcher memo generation failed; default memo used",
            exc,
            {"chapter": chapter_name},
        )
        return chapter_memory or default_style_memo()


_HALLUCINATION_RE = re.compile(
    r"\s*\(\s*\d+\s+s[eé]ance[s]?\s*\)"  # (1 séance), (2 seances) etc.
    r"|\s*\(\s*session\s+\d+\s*\)"        # (session 1)
    r"|\s*\(\s*chapitre\s+\d+\s*\)",      # (chapitre 1)
    flags=re.IGNORECASE,
)


def clean_generated_text(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:text|fr|french)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    for prefix in ("Corrected French:", "Repaired French:", "French:", "Traduction:"):
        if value.lower().startswith(prefix.lower()):
            value = value[len(prefix):].strip()
    # Strip common MADLAD hallucinations appended to chapter headers/datelines.
    value = _HALLUCINATION_RE.sub("", value)
    return value.strip()


def issues_from_flags(flags: list[str]) -> str:
    return "\n".join(f"- validator:{flag}" for flag in flags)


def format_alt_drafts(record: ChunkRecord) -> str:
    if not record.alt_drafts:
        return ""
    lines = []
    for model_name, text in sorted(record.alt_drafts.items()):
        if text:
            lines.append(f"{model_name}: {text}")
    return "\n".join(lines)
