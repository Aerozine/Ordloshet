"""Translation quality validators and inline marker repair."""
from __future__ import annotations
import re, collections.abc
from lib.models import ChunkRecord
from lib.char_graph import (
    CHARACTER_ALIASES, strip_inline_markers, source_aliases,
    has_direct_speech, likely_present_tense_drift,
    SECOND_PERSON_SOURCE_RE, PLURAL_OR_FORMAL_SOURCE_RE,
    FRENCH_DIALOGUE_MARKER_RE, source_prefers_informal_second_person,
)
from lib.constants import INLINE_MARKER_HINTS, ENGLISH_MARKERS, FRENCH_MARKERS

GLOSSARY_RULES: dict[str, dict] = {}

def validate_style_rules(record: ChunkRecord, candidate: str) -> list[str]:
    source = strip_inline_markers(record.source)
    target = strip_inline_markers(candidate)
    flags: list[str] = []
    hints = set(record.style_hints or [])

    if "dialogue" in hints and not FRENCH_DIALOGUE_MARKER_RE.search(target):
        flags.append("dialogue_punctuation_lost")

    if "preserve_question" in hints and "?" not in target:
        flags.append("question_punctuation_lost")

    if "past_narration" in hints and likely_present_tense_drift(target):
        flags.append("tense_present_drift")

    canonical_names = set(CHARACTER_ALIASES.values())
    for alias in source_aliases(source):
        if alias in canonical_names:
            continue
        if not re.search(rf"\b{re.escape(alias)}\b", target):
            flags.append(f"address_name_lost_{alias}")

    source_mentions = set(record.mentioned_characters or [])
    if record.speaker:
        source_mentions.add(record.speaker)
    for character in source_mentions:
        if character in canonical_names and character in source:
            if not re.search(rf"\b{re.escape(character)}\b", target):
                flags.append(f"scene_name_lost_{character}")

    return flags


def text_matches_any(value: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def glossary_slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")


def source_has_glossary_term(source: str, rule: dict[str, list[str]]) -> bool:
    return text_matches_any(strip_inline_markers(source), rule.get("source", []))


def validate_glossary_rules(record: ChunkRecord, candidate: str) -> list[str]:
    if not getattr(validate_glossary_rules, "_enabled", True):
        return []
    flags: list[str] = []
    target = strip_inline_markers(candidate)
    for term, rule in GLOSSARY_RULES.items():
        if not source_has_glossary_term(record.source, rule):
            continue
        slug = glossary_slug(term)
        preferred = rule.get("preferred", [])
        forbidden = rule.get("forbidden", [])
        if preferred and not text_matches_any(target, preferred):
            flags.append(f"glossary_missing_{slug}")
        if forbidden and text_matches_any(target, forbidden):
            flags.append(f"glossary_forbidden_{slug}")
    return flags


def validate_record_translation(
    record: ChunkRecord,
    draft: str,
    candidate: str,
) -> list[str]:
    flags = validate_translation(record.source, draft, candidate, record.unit.markers)
    if record.register_hint == "plural_or_formal_allowed":
        flags = [
            flag for flag in flags
            if flag not in {"wrong_register_vous", "unexpected_vous"}
        ]
    elif record.register_hint == "informal" and source_prefers_informal_second_person(candidate):
        if "wrong_register_vous" not in flags:
            flags.append("informal_register_hint_present")
    flags.extend(validate_style_rules(record, candidate))
    flags.extend(validate_glossary_rules(record, candidate))

    # Lightweight regex-based French quality checks (no extra deps)
    try:
        from lib.french_nlp import (
            detect_english_leaks,
            detect_hallucinated_numbers,
            check_length_ratio,
        )
        for leaked in detect_english_leaks(record.source, candidate)[:3]:
            flags.append(f"english_leak_{leaked.lower()}")
        for halluc in detect_hallucinated_numbers(record.source, candidate)[:2]:
            flags.append(halluc)
        ratio_warn = check_length_ratio(record.source, candidate)
        if ratio_warn and "possible_omission" in ratio_warn:
            flags.append("possible_omission")
        from lib.french_nlp import detect_dialogue_english
        for _ in detect_dialogue_english(candidate, record.source)[:2]:
            flags.append("untranslated_dialogue")
    except Exception:
        pass

    # spaCy-based checks (enabled via --french-nlp)
    if getattr(validate_record_translation, "_french_nlp_enabled", False):
        try:
            from lib.french_nlp import detect_tense_mixing, detect_agreement_errors
            if detect_tense_mixing(candidate):
                flags.append("tense_mixing")
            flags.extend(detect_agreement_errors(candidate, max_errors=2))
        except Exception:
            pass

    return list(dict.fromkeys(flags))


def source_names(source: str, extra_names: "collection.abc.Iterable[str] | None" = None) -> list[str]:
    """Return proper names present in source.

    Uses the module-level _SOURCE_NAMES_REGISTRY if populated (filled from
    entity memory at chapter start), falling back to CHARACTER_ALIASES values.
    """
    registry = getattr(source_names, "_registry", None)
    candidates = list(registry) if registry else list(set(CHARACTER_ALIASES.values()))
    if extra_names:
        for n in extra_names:
            if n not in candidates:
                candidates.append(n)
    return [name for name in candidates if re.search(rf"\b{re.escape(name)}\b", source)]


def register_source_names(names: "list[str]") -> None:
    """Call once per chapter to set the active name registry for source_names()."""
    source_names._registry = list(dict.fromkeys(n for n in names if n and n[0].isupper()))


def _register_chapter_names(records: "list[ChunkRecord]", args: "argparse.Namespace") -> None:
    """Extract proper names from chapter source text and register them globally."""
    all_source = " ".join(r.source for r in records)
    # Capitalized words that appear at least twice are likely proper names.
    candidates = re.findall(r"\b([A-Z][a-z]{1,})\b", all_source)
    freq: dict[str, int] = {}
    for c in candidates:
        freq[c] = freq.get(c, 0) + 1
    # Also pull from glossary (keys that start with a capital letter).
    glossary_names = [
        k.strip().title() for k in getattr(args, "_active_glossary", {})
        if k and k[0].isupper()
    ]
    names = [n for n, count in freq.items() if count >= 2] + glossary_names
    # De-duplicate and sort by frequency so most common names come first.
    names = sorted(set(names), key=lambda n: -freq.get(n, 0))
    register_source_names(names[:40])


def marker_content_by_id(marked_source: str) -> dict[str, str]:
    result: dict[str, str] = {}
    pattern = re.compile(r"\[\[(ZX[A-Z]+[0-9]+X)\]\](.*?)\[\[/\1\]\]", re.DOTALL)
    for marker_id, content in pattern.findall(marked_source):
        result[marker_id] = content.strip()
    return result


def repair_inline_markers_from_hints(
    source: str,
    candidate: str,
    markers: list[InlineMarker],
) -> str:
    missing = missing_markers(candidate, markers)
    if not missing:
        return candidate

    repaired = candidate
    source_content = marker_content_by_id(source)
    for marker_id in missing:
        raw_hint = source_content.get(marker_id, "")
        normalized_hint = re.sub(r"\s+", " ", raw_hint.lower()).strip(" .?!,;:")
        hint_patterns = INLINE_MARKER_HINTS.get(normalized_hint, [])
        placed = False
        for pattern in hint_patterns:
            match = re.search(pattern, repaired, flags=re.IGNORECASE)
            if not match:
                continue
            repaired = (
                repaired[:match.start()]
                + f"[[{marker_id}]]"
                + repaired[match.start():match.end()]
                + f"[[/{marker_id}]]"
                + repaired[match.end():]
            )
            placed = True
            break
        if not placed:
            # Hard fallback: append the marker wrapping its original source
            # content so downstream EPUB assembly can place it correctly.
            content = raw_hint or marker_id
            repaired = repaired.rstrip() + f" [[{marker_id}]]{content}[[/{marker_id}]]"
    return repaired


def validate_translation(
    source: str,
    draft: str,
    candidate: str,
    markers: list[InlineMarker],
) -> list[str]:
    flags: list[str] = []
    stripped = candidate.strip()
    if not stripped:
        flags.append("empty")

    if any(marker.lower() in stripped.lower() for marker in PROMPT_LEAK_MARKERS):
        flags.append("prompt_leak")

    tokens = word_tokens(stripped)
    if tokens:
        source_tokens = set(word_tokens(source))
        # Don't count English words that were already in the source — they are
        # proper nouns, sport terms, or intentionally preserved English that the
        # model correctly kept as-is.
        english_hits = sum(
            token in ENGLISH_MARKERS and token not in source_tokens
            for token in tokens
        )
        french_hits = sum(token in FRENCH_MARKERS for token in tokens)
        if english_hits >= 4 and english_hits > french_hits:
            flags.append("english_heavy")

    baseline_len = max(len(draft.strip()), len(source.strip()), 1)
    ratio = len(stripped) / baseline_len
    if baseline_len > 40 and (ratio > 2.2 or ratio < 0.35):
        flags.append("length_ratio")

    if missing_markers(stripped, markers):
        flags.append("missing_inline_markers")

    if has_direct_speech(source) and '"' in stripped:
        flags.append("ascii_dialogue_quotes")

    for name in source_names(source):
        if not re.search(rf"\b{name}\b", stripped):
            flags.append(f"missing_name_{name}")

    return flags
