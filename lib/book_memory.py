"""Per-book automatic glossary/style/register memory."""
from __future__ import annotations
import json
from pathlib import Path
from lib.models import ChunkRecord
from lib.char_graph import CHARACTER_ALIASES, CHARACTER_GENDERS, CHARACTER_RELATIONSHIPS
from lib.logging_utils import write_progress
from lib.registries import DEFAULT_BOOK_MEMORY_DIR

def book_memory_path(args: argparse.Namespace) -> Path:
    book_id = str(getattr(args, "_current_book_stem", "book") or "book")
    return Path(args.book_memory_dir) / f"{safe_output_stem(book_id)}.json"


def load_book_memory(args: argparse.Namespace) -> dict[str, object]:
    if not getattr(args, "book_memory", False):
        return {}
    cached = getattr(load_book_memory, "_cache", None)
    cached_path = getattr(load_book_memory, "_path", None)
    path = book_memory_path(args)
    if cached is not None and cached_path == str(path):
        return cached
    if not path.is_file():
        memory: dict[str, object] = {
            "version": 1,
            "chapters": [],
            "characters": {},
            "glossary_terms": {},
            "register_edges": {},
            "style_rules": [],
        }
        setattr(load_book_memory, "_cache", memory)
        setattr(load_book_memory, "_path", str(path))
        return memory
    try:
        with path.open("r", encoding="utf-8") as handle:
            memory = json.load(handle)
        if not isinstance(memory, dict):
            memory = {}
    except Exception as exc:
        log_error(args, "Book memory read failed", exc, {"path": path})
        memory = {}
    memory.setdefault("version", 1)
    memory.setdefault("chapters", [])
    memory.setdefault("characters", {})
    memory.setdefault("glossary_terms", {})
    memory.setdefault("register_edges", {})
    memory.setdefault("style_rules", [])
    setattr(load_book_memory, "_cache", memory)
    setattr(load_book_memory, "_path", str(path))
    return memory


def book_memory_summary(args: argparse.Namespace) -> str:
    memory = load_book_memory(args)
    if not memory:
        return ""
    lines = ["Book memory:"]
    glossary_terms = memory.get("glossary_terms", {})
    if isinstance(glossary_terms, dict) and glossary_terms:
        lines.append("Persistent glossary:")
        for term, value in sorted(glossary_terms.items()):
            if isinstance(value, dict):
                preferred = value.get("preferred", "")
                count = value.get("count", 0)
                lines.append(f"- {term} -> {preferred} (seen {count})")
    register_edges = memory.get("register_edges", {})
    if isinstance(register_edges, dict) and register_edges:
        lines.append("Persistent register decisions:")
        for pair, value in sorted(register_edges.items()):
            lines.append(f"- {pair}: {value}")
    style_rules = memory.get("style_rules", [])
    if isinstance(style_rules, list) and style_rules:
        lines.append("Persistent style rules:")
        lines.extend(f"- {rule}" for rule in style_rules[:12])
    return "\n".join(lines)


def apply_book_register_memory(records: list[ChunkRecord], args: argparse.Namespace) -> None:
    if not getattr(args, "book_memory", False):
        return
    memory = load_book_memory(args)
    register_edges = memory.get("register_edges", {})
    if not isinstance(register_edges, dict) or not register_edges:
        return
    changed = 0
    for record in records:
        if record.register_hint or not record.speaker or not record.addressee:
            continue
        text = strip_inline_markers(record.source)
        if not SECOND_PERSON_SOURCE_RE.search(text) or PLURAL_OR_FORMAL_SOURCE_RE.search(text):
            continue
        hint = register_edges.get(f"{record.speaker}->{record.addressee}")
        if hint is None:
            hint = register_edges.get(f"{record.addressee}->{record.speaker}")
        if hint in {"informal", "plural_or_formal_allowed"}:
            record.register_hint = str(hint)
            record.style_hints = style_hints_for_record(record)
            changed += 1
    if changed:
        write_progress(f"Book memory: applied {changed} register hint(s).")


def save_book_memory(args: argparse.Namespace, records: list[ChunkRecord], chapter_name: str) -> None:
    if not getattr(args, "book_memory", False) or args.dry_run:
        return
    memory = load_book_memory(args)
    chapters = memory.setdefault("chapters", [])
    if isinstance(chapters, list) and chapter_name not in chapters:
        chapters.append(chapter_name)
    characters = memory.setdefault("characters", {})
    if isinstance(characters, dict):
        for record in records:
            for character in record.mentioned_characters or []:
                item = characters.setdefault(
                    character,
                    {"count": 0, "gender": CHARACTER_GENDERS.get(character, "")},
                )
                if isinstance(item, dict):
                    item["count"] = int(item.get("count", 0)) + 1
    glossary_terms = memory.setdefault("glossary_terms", {})
    if isinstance(glossary_terms, dict):
        for term, rule in GLOSSARY_RULES.items():
            count = sum(1 for record in records if source_has_glossary_term(record.source, rule))
            if not count:
                continue
            item = glossary_terms.setdefault(
                term,
                {"preferred": KNOWN_TRANSLATION_TERMS.get(term, ""), "count": 0},
            )
            if isinstance(item, dict):
                item["count"] = int(item.get("count", 0)) + count
    register_edges = memory.setdefault("register_edges", {})
    if isinstance(register_edges, dict):
        for record in records:
            if record.speaker and record.addressee and record.register_hint:
                register_edges[f"{record.speaker}->{record.addressee}"] = record.register_hint
    style_rules = memory.setdefault("style_rules", [])
    if isinstance(style_rules, list):
        for rule in build_chapter_entity_graph(records, chapter_name).style_rules:
            if rule not in style_rules:
                style_rules.append(rule)
    path = book_memory_path(args)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(memory, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temp_path.replace(path)
    except Exception as exc:
        log_error(args, "Book memory write failed", exc, {"path": path})
