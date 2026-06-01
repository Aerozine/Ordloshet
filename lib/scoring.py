"""Candidate scoring, acceptance logic, and reranking."""
from __future__ import annotations
import re
from lib.models import ChunkRecord, InlineMarker, CandidateChoice
from lib.char_graph import strip_inline_markers, CHARACTER_ALIASES, character_mentions
from lib.validators import validate_style_rules, validate_glossary_rules, GLOSSARY_RULES
from lib.logging_utils import write_progress

def populate_arbitration_drafts(
    decisions: list[PatchDecision],
    args: argparse.Namespace,
    chapter_name: str,
) -> None:
    if (
        args.dry_run
        or not getattr(args, "literary_arbitration", False)
        or not decisions
    ):
        return

    model_name = str(getattr(args, "arbitration_model", "nllb-ct2"))
    if not model_name:
        return

    selected = [decision.record for decision in decisions if decision.needs_patch]
    max_chunks = max(0, int(getattr(args, "arbitration_max_chunks", 0)))
    if max_chunks:
        selected = selected[:max_chunks]
    if not selected:
        return

    write_progress(
        f"Arbitration: translating {len(selected)} risky chunk(s) with {model_name}."
    )
    handle = load_translator(model_name)
    try:
        if not args.dry_run:
            handle.preload()
        outputs = translate_records(
            selected,
            handle,
            args,
            f"Arbitration {model_name}",
            chapter_name,
        )
        for record, output in zip(selected, outputs):
            if record.alt_drafts is None:
                record.alt_drafts = {}
            record.alt_drafts[model_name] = output
    except Exception as exc:
        log_error(
            args,
            "Arbitration draft failed; continuing without secondary draft",
            exc,
            {"chapter": chapter_name, "model": model_name},
        )
        write_progress(f"Arbitration draft failed for {model_name}: {short_error(exc)}")
    finally:
        handle.cleanup()


_HARD_VALIDATION_FLAGS_STATIC = {
    "empty",
    "prompt_leak",
    "english_heavy",
    "length_ratio",
    "missing_inline_markers",
    "wrong_register_vous",
    "unexpected_vous",
    "negation_risk",
    "profanity_softened",
    "dialogue_punctuation_lost",
    "question_punctuation_lost",
    "tense_present_drift",
}

_HARD_FLAG_PREFIXES = (
    "missing_name_",
    "address_name_lost_",
    "scene_name_lost_",
    "glossary_missing_",
    "glossary_forbidden_",
)


def hard_flag_count(flags: list[str]) -> int:
    return sum(
        1 for flag in flags
        if flag in _HARD_VALIDATION_FLAGS_STATIC
        or any(flag.startswith(p) for p in _HARD_FLAG_PREFIXES)
    )


HARD_VALIDATION_FLAGS = _HARD_VALIDATION_FLAGS_STATIC  # backward-compat alias


def candidate_accepted(
    record: ChunkRecord,
    baseline: str,
    candidate: str,
    args: argparse.Namespace,
) -> tuple[bool, list[str], float, float]:
    candidate_flags = validate_record_translation(record, record.draft, candidate)
    baseline_flags = validate_record_translation(record, record.draft, baseline)
    candidate_score_value = candidate_score_for_record(record, candidate)
    baseline_score_value = candidate_score_for_record(record, baseline)

    if not getattr(args, "strict_acceptance", True):
        return (
            candidate_score_value + args.patch_accept_margin >= baseline_score_value,
            candidate_flags,
            candidate_score_value,
            baseline_score_value,
        )

    candidate_hard = hard_flag_count(candidate_flags)
    baseline_hard = hard_flag_count(baseline_flags)
    if candidate_hard > baseline_hard:
        return False, candidate_flags, candidate_score_value, baseline_score_value
    if candidate_hard and candidate_hard == baseline_hard:
        return False, candidate_flags, candidate_score_value, baseline_score_value
    if candidate_score_value + args.patch_accept_margin < baseline_score_value:
        return False, candidate_flags, candidate_score_value, baseline_score_value
    return True, candidate_flags, candidate_score_value, baseline_score_value


def rerank_record_candidates(
    record: ChunkRecord,
    candidates: dict[str, str],
    args: argparse.Namespace,
) -> tuple[str, str, list[str], float]:
    if not candidates:
        return "draft", record.draft, validate_record_translation(record, record.draft, record.draft), candidate_score_for_record(record, record.draft)

    best_name = ""
    best_text = ""
    best_flags: list[str] = []
    best_score = -1_000_000.0
    best_hard = 1_000_000
    for name, text in candidates.items():
        if not text:
            continue
        flags = validate_record_translation(record, record.draft, text)
        hard = hard_flag_count(flags)
        score = candidate_score_for_record(record, text)
        if getattr(args, "candidate_reranking", True):
            if hard < best_hard or (hard == best_hard and score > best_score):
                best_name = name
                best_text = text
                best_flags = flags
                best_score = score
                best_hard = hard
        elif name == "patch":
            return name, text, flags, score

    if not best_text:
        best_text = record.draft
        best_name = "draft"
        best_flags = validate_record_translation(record, record.draft, best_text)
        best_score = candidate_score_for_record(record, best_text)
    return best_name, best_text, best_flags, best_score


def candidate_score(source: str, candidate: str, markers: list[InlineMarker]) -> float:
    flags = validate_translation(source, source, candidate, markers)
    score = 100.0
    for flag in flags:
        if flag in {"wrong_register_vous", "unexpected_vous", "missing_inline_markers", "prompt_leak"}:
            score -= 45.0
        elif (
            flag in {"negation_risk", "dialogue_punctuation_lost", "tense_present_drift"}
            or flag.startswith("missing_name_")
        ):
            score -= 30.0
        elif (
            flag in {"english_heavy", "length_ratio", "profanity_softened", "question_punctuation_lost"}
            or flag.startswith("address_name_lost_")
            or flag.startswith("scene_name_lost_")
        ):
            score -= 25.0
        else:
            score -= 12.0
    tokens = word_tokens(candidate)
    if tokens:
        source_tokens = set(word_tokens(source))
        english_hits = sum(
            token in ENGLISH_MARKERS and token not in source_tokens for token in tokens
        )
        french_hits = sum(token in FRENCH_MARKERS for token in tokens)
        score += min(25.0, french_hits * 0.4)
        score -= min(25.0, english_hits * 0.8)
    if any(ord(char) > 127 for char in candidate):
        score += 4.0
    ratio = len(candidate.strip()) / max(1, len(source.strip()))
    if 0.8 <= ratio <= 1.8:
        score += 5.0
    return score


def score_from_flags(candidate: str, source: str, flags: list[str]) -> float:
    score = 100.0
    for flag in flags:
        if flag in {"wrong_register_vous", "unexpected_vous", "missing_inline_markers", "prompt_leak"}:
            score -= 45.0
        elif (
            flag in {"negation_risk", "dialogue_punctuation_lost", "tense_present_drift"}
            or flag.startswith("missing_name_")
        ):
            score -= 30.0
        elif (
            flag in {"english_heavy", "length_ratio", "profanity_softened", "question_punctuation_lost"}
            or flag.startswith("address_name_lost_")
            or flag.startswith("scene_name_lost_")
        ):
            score -= 25.0
        else:
            score -= 12.0
    tokens = word_tokens(candidate)
    if tokens:
        source_tokens = set(word_tokens(source))
        english_hits = sum(
            token in ENGLISH_MARKERS and token not in source_tokens for token in tokens
        )
        french_hits = sum(token in FRENCH_MARKERS for token in tokens)
        score += min(25.0, french_hits * 0.4)
        score -= min(25.0, english_hits * 0.8)
    if any(ord(char) > 127 for char in candidate):
        score += 4.0
    ratio = len(candidate.strip()) / max(1, len(source.strip()))
    if 0.8 <= ratio <= 1.8:
        score += 5.0
    return score


def candidate_score_for_record(record: ChunkRecord, candidate: str, draft: str | None = None) -> float:
    baseline = record.draft if draft is None else draft
    flags = validate_record_translation(record, baseline, candidate)
    return score_from_flags(candidate, record.source, flags)
