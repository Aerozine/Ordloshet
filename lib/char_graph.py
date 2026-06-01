"""Character relationship graph and dialogue analysis.

This module owns the mutable character-data globals (CHARACTER_ALIASES,
CHARACTER_GENDERS, CHARACTER_RELATIONSHIPS) so that lib/book_config.py and
lib/book_analyzer.py can patch them without importing epub.py.

epub.py re-imports these names so that in-place dict mutations are reflected
everywhere:
    from lib.char_graph import CHARACTER_ALIASES, CHARACTER_GENDERS, CHARACTER_RELATIONSHIPS
"""
from __future__ import annotations

import re
from typing import Iterable

from lib.models import ChunkRecord, ChapterEntityGraph

# --- Mutable character data (populated by book_config / book_analyzer) ------

CHARACTER_ALIASES: dict[str, str] = {}
CHARACTER_GENDERS: dict[str, str] = {}
CHARACTER_RELATIONSHIPS: dict[str, str] = {}

# --- Regex helpers (language-independent) -----------------------------------

SPEECH_VERBS_RE = re.compile(
    r"\b(said|asked|called|shouted|whispered|muttered|snapped|replied|answered|"
    r"laughed|continued|told|yelled|groaned|huffed|wheezed)\b",
    re.IGNORECASE,
)
SOURCE_PAST_RE = re.compile(
    r"\b(was|were|had|felt|looked|smiled|laughed|said|asked|called|reached|"
    r"collapsed|stumbled|put|waited|opened|pulled|closed|tilted|tried|"
    r"grabbed|rolled|dropped|sprang|gave|wanted)\b",
    re.IGNORECASE,
)
# Matches French present-tense verbs after a pronoun — signals tense drift.
# Character names are NOT hardcoded here; only generic subject pronouns.
FRENCH_PRESENT_DRIFT_RE = re.compile(
    r"\b(?:il|elle|ils|elles)\s+"
    r"(?:s'essuie|essuie|tient|regarde|sourit|rit|ouvre|ferme|"
    r"demande|repond|répond|dit|pense|sait|veut|peut|attrape|tombe|"
    r"se lève|lève|marche|court|parle|crie|chuchote)\b",
    re.IGNORECASE,
)
FRENCH_DIALOGUE_MARKER_RE = re.compile(r"(^|\s)(?:[\"«»]|[-–—]\s)")

# English source-language patterns for register inference.
# These detect properties of the SOURCE text only — no target-language content.
SECOND_PERSON_SOURCE_RE = re.compile(
    r"\b(you|your|yours|yourself|you're|you've|you'll|you'd)\b",
    re.IGNORECASE,
)
PLURAL_OR_FORMAL_SOURCE_RE = re.compile(
    r"\b(you guys|both of you|all of you|you two|the two of you|"
    r"with both of you|for both of you|between you|your side)\b",
    re.IGNORECASE,
)


# --- Utility ----------------------------------------------------------------

def strip_inline_markers(value: str) -> str:
    return re.sub(r"\[\[/?ZX[A-Z]+[0-9]+X\]\]", "", value)


def unique_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


# --- Character helpers -------------------------------------------------------

def canonical_character_name(value: str) -> str:
    return CHARACTER_ALIASES.get(value, value)


def character_pair_key(left: str, right: str) -> str:
    return f"{canonical_character_name(left)}|{canonical_character_name(right)}"


def relationship_between(left: str, right: str) -> str:
    if not left or not right:
        return ""
    return CHARACTER_RELATIONSHIPS.get(character_pair_key(left, right), "")


def intimate_partner_of(name: str) -> str:
    """Return the intimate partner of *name* from CHARACTER_RELATIONSHIPS, or ''."""
    canonical = canonical_character_name(name)
    for key, rel in CHARACTER_RELATIONSHIPS.items():
        if rel != "intimate_partner":
            continue
        parts = key.split("|", 1)
        if len(parts) == 2 and parts[0] == canonical:
            return parts[1]
    return ""


def intimate_pairs() -> list[frozenset[str]]:
    """Return deduplicated frozensets of all intimate partner pairs."""
    seen: set[frozenset] = set()
    result = []
    for key, rel in CHARACTER_RELATIONSHIPS.items():
        if rel != "intimate_partner":
            continue
        parts = key.split("|", 1)
        if len(parts) == 2:
            fs = frozenset(parts)
            if fs not in seen:
                seen.add(fs)
                result.append(fs)
    return result


def character_mentions(source: str) -> list[str]:
    text = strip_inline_markers(source)
    mentions = []
    for alias, canonical in CHARACTER_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            mentions.append(canonical)
    return unique_keep_order(mentions)


def source_aliases(source: str) -> list[str]:
    text = strip_inline_markers(source)
    return [
        alias for alias in CHARACTER_ALIASES
        if re.search(rf"\b{re.escape(alias)}\b", text)
    ]


def has_direct_speech(value: str) -> bool:
    return any(marker in value for marker in ('"', "“", "”", "«", "»"))


def likely_past_narration(source: str) -> bool:
    text = strip_inline_markers(source)
    return not has_direct_speech(text) and SOURCE_PAST_RE.search(text) is not None


def likely_present_tense_drift(candidate: str) -> bool:
    return FRENCH_PRESENT_DRIFT_RE.search(strip_inline_markers(candidate)) is not None



def text_without_quoted_dialogue(source: str) -> str:
    text = strip_inline_markers(source)
    text = re.sub(r'"[^"]*"', " ", text)
    text = re.sub(r"“[^”]*”", " ", text)
    text = re.sub(r"«[^»]*»", " ", text)
    return text


def source_prefers_informal_second_person(source: str) -> bool:
    text = strip_inline_markers(source)
    if not has_direct_speech(text):
        return False
    if not SECOND_PERSON_SOURCE_RE.search(text):
        return False
    if PLURAL_OR_FORMAL_SOURCE_RE.search(text):
        return False
    # If the detected speaker is NOT part of any intimate pair, don't force
    # informal — they may legitimately use formal register.
    speaker = infer_source_speaker(source)
    if speaker and not intimate_partner_of(speaker) and not any(speaker in p for p in intimate_pairs()):
        return False
    return True


def infer_source_speaker(source: str) -> str:
    text = text_without_quoted_dialogue(source)
    for alias, canonical in CHARACTER_ALIASES.items():
        escaped = re.escape(alias)
        if re.search(rf"\b{escaped}\b[^.\n]{{0,80}}{SPEECH_VERBS_RE.pattern}", text, re.IGNORECASE):
            return canonical
        if re.search(rf"{SPEECH_VERBS_RE.pattern}[^.\n]{{0,80}}\b{escaped}\b", text, re.IGNORECASE):
            return canonical
    return ""


def infer_source_addressee(source: str, speaker: str) -> str:
    """Infer the addressee of a dialogue chunk using CHARACTER_RELATIONSHIPS."""
    text = strip_inline_markers(source)

    # Named mention implies that character is being addressed.
    if speaker:
        for alias, canonical in CHARACTER_ALIASES.items():
            if canonical == speaker:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return canonical

    # Speaker's intimate partner is the default addressee in dialogue.
    if speaker and has_direct_speech(text):
        partner = intimate_partner_of(speaker)
        if partner:
            return partner

    # Alias mentions in dialogue without a known speaker.
    for alias, canonical in CHARACTER_ALIASES.items():
        if alias != canonical and re.search(rf"\b{re.escape(alias)}\b", text) and has_direct_speech(text):
            return canonical

    return ""


def other_recent_participant(speaker: str, recent_participants: list[str]) -> str:
    for participant in recent_participants:
        if participant != speaker:
            return participant
    return intimate_partner_of(speaker)


def infer_unknown_dialogue_speaker(
    mentions: list[str],
    last_speaker: str,
    recent_participants: list[str],
) -> str:
    all_intimate = {name for pair in intimate_pairs() for name in pair}
    if len(mentions) == 1 and mentions[0] in all_intimate:
        return other_recent_participant(mentions[0], list(all_intimate))
    if last_speaker in all_intimate:
        return other_recent_participant(last_speaker, list(all_intimate))
    if len(recent_participants) == 2:
        return recent_participants[0]
    return ""


def style_hints_for_record(record: ChunkRecord) -> list[str]:
    hints = []
    source = strip_inline_markers(record.source)
    if has_direct_speech(source):
        hints.append("dialogue")
    if record.register_hint == "informal":
        hints.append("informal_dialogue_required")
    if record.register_hint == "plural_or_formal_allowed":
        hints.append("plural_or_formal_allowed")
    if likely_past_narration(source):
        hints.append("past_narration")
    if source_aliases(source):
        hints.append("preserve_address_names")
    if "?" in source:
        hints.append("preserve_question")
    return hints


def _generate_style_rules() -> list[str]:
    """Build dynamic style rules from CHARACTER_RELATIONSHIPS."""
    rules = []
    for pair in intimate_pairs():
        names = sorted(pair)
        rules.append(
            f"{' and '.join(names)} direct dialogue must use informal tu/toi/te/ton/ta/tes."
        )
    rules += [
        "Preserve all address names and aliases exactly.",
        "Preserve direct-speech punctuation; do not turn dialogue into narration.",
        "Keep past-tense narration consistent; avoid present-tense drift.",
        "Preserve questions, negation, names, profanity strength, and inline markers.",
    ]
    return rules


def build_chapter_entity_graph(
    records: list[ChunkRecord], chapter_name: str
) -> ChapterEntityGraph:
    last_pair_speaker = ""
    recent_participants: list[str] = []
    dialogue_edges: list[dict[str, object]] = []
    scene_participants_by_chunk: dict[int, list[str]] = {}
    ambiguous_dialogue_chunks: list[int] = []

    all_intimate_names = {name for pair in intimate_pairs() for name in pair}

    for record in records:
        text = strip_inline_markers(record.source)
        mentions = character_mentions(text)
        if mentions:
            recent_participants = unique_keep_order(mentions + recent_participants)[:4]

        speaker = infer_source_speaker(text)
        if not speaker and has_direct_speech(text):
            speaker = infer_unknown_dialogue_speaker(
                mentions, last_pair_speaker, recent_participants
            )
            if not speaker:
                ambiguous_dialogue_chunks.append(record.index)

        addressee = infer_source_addressee(text, speaker)

        if speaker in all_intimate_names:
            last_pair_speaker = speaker
        elif has_direct_speech(text) and not speaker and last_pair_speaker:
            addressee = intimate_partner_of(last_pair_speaker) or addressee

        if speaker and not addressee and has_direct_speech(text):
            addressee = other_recent_participant(speaker, recent_participants)

        register_hint = ""
        if PLURAL_OR_FORMAL_SOURCE_RE.search(text):
            register_hint = "plural_or_formal_allowed"
        elif (
            source_prefers_informal_second_person(text)
            or (
                has_direct_speech(text)
                and (
                    relationship_between(speaker, addressee) == "intimate_partner"
                    or (speaker in all_intimate_names and addressee in all_intimate_names)
                )
            )
        ):
            register_hint = "informal"
        elif (
            not register_hint
            and getattr(build_chapter_entity_graph, "_force_tu", False)
            and SECOND_PERSON_SOURCE_RE.search(text)
            and not PLURAL_OR_FORMAL_SOURCE_RE.search(text)
        ):
            register_hint = "informal"

        participants = unique_keep_order(
            mentions + [speaker, addressee] + recent_participants
        )
        participants = [p for p in participants if p][:4]
        record.speaker = speaker
        record.addressee = addressee
        record.register_hint = register_hint
        record.mentioned_characters = mentions
        record.scene_participants = participants
        record.style_hints = style_hints_for_record(record)

        if speaker or addressee:
            dialogue_edges.append({
                "chunk": record.index,
                "speaker": speaker or "unknown",
                "addressee": addressee or "unknown",
                "register": register_hint or "default",
            })
        scene_participants_by_chunk[record.index] = participants

    characters = sorted({
        character
        for record in records
        for character in (record.mentioned_characters or [])
    })

    return ChapterEntityGraph(
        chapter_name=chapter_name,
        characters=characters,
        aliases=dict(CHARACTER_ALIASES),
        relationships=dict(CHARACTER_RELATIONSHIPS),
        style_rules=_generate_style_rules(),
        dialogue_edges=dialogue_edges,
        scene_participants_by_chunk=scene_participants_by_chunk,
        ambiguous_dialogue_chunks=ambiguous_dialogue_chunks,
    )


def patch_character_data(
    aliases: dict[str, str] | None = None,
    genders: dict[str, str] | None = None,
    relationships: dict[str, str] | None = None,
    glossary: dict | None = None,
    force_tu: bool = False,
) -> None:
    """Update the mutable character globals in-place (called by book_config / book_analyzer)."""
    if aliases is not None:
        CHARACTER_ALIASES.clear()
        CHARACTER_ALIASES.update(aliases)
    if genders is not None:
        CHARACTER_GENDERS.clear()
        CHARACTER_GENDERS.update(genders)
    if relationships is not None:
        CHARACTER_RELATIONSHIPS.clear()
        CHARACTER_RELATIONSHIPS.update(relationships)
    build_chapter_entity_graph._force_tu = force_tu  # type: ignore[attr-defined]
    if glossary is not None:
        try:
            from lib import validators as _val
            _val.GLOSSARY_RULES.clear()
            _val.GLOSSARY_RULES.update(glossary)
        except Exception:
            pass
