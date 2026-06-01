"""Translation quality evaluation (chrF-like scoring, evaluation reports)."""
from __future__ import annotations
import json, re
from pathlib import Path
from lib.models import ChunkRecord
from lib.char_graph import strip_inline_markers
from lib.logging_utils import write_progress
from lib.registries import DEFAULT_EVALUATION_DIR

def chapter_plain_text_from_records(records: list[ChunkRecord]) -> str:
    return "\n".join(strip_inline_markers(record.final or record.draft) for record in records)


def character_ngrams(value: str, width: int) -> dict[str, int]:
    text = re.sub(r"\s+", " ", value.lower()).strip()
    if len(text) < width:
        return {text: 1} if text else {}
    counts: dict[str, int] = {}
    for index in range(len(text) - width + 1):
        gram = text[index:index + width]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def chrf_like(candidate: str, reference: str, max_order: int = 6, beta: float = 2.0) -> float:
    if not candidate.strip() or not reference.strip():
        return 0.0
    scores = []
    for width in range(1, max_order + 1):
        cand = character_ngrams(candidate, width)
        ref = character_ngrams(reference, width)
        overlap = sum(min(count, ref.get(gram, 0)) for gram, count in cand.items())
        precision = overlap / max(1, sum(cand.values()))
        recall = overlap / max(1, sum(ref.values()))
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((1 + beta * beta) * precision * recall / (beta * beta * precision + recall))
    return round(100.0 * sum(scores) / len(scores), 2)


def strip_html_for_reference(raw: str) -> str:
    import html as html_lib

    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.I)
    raw = re.sub(r"</(p|div|h\d|li)>", "\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html_lib.unescape(raw).replace("\u00a0", " ")
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s+", "\n", raw)
    raw = re.sub(r"\n{2,}", "\n", raw)
    return raw.strip()


def reference_chapter_text(args: argparse.Namespace) -> str:
    reference_path = Path(args.eval_reference) if args.eval_reference else None
    if reference_path is None or not reference_path.is_file():
        return ""

    chapter_number = requested_chapter_number(args.chapter) or 1
    preferred_suffix = f"ch{chapter_number + 1:03d}.xhtml"
    try:
        import zipfile

        with zipfile.ZipFile(reference_path) as archive:
            names = [
                name for name in archive.namelist()
                if name.lower().endswith((".xhtml", ".html"))
            ]
            for name in names:
                if name.endswith(preferred_suffix):
                    return strip_html_for_reference(archive.read(name).decode("utf-8", "ignore"))
            chapter_markers = (
                f"chapitre {chapter_number}",
                f"chapitre {number_to_words(chapter_number) or chapter_number}",
            )
            for name in names:
                text = strip_html_for_reference(archive.read(name).decode("utf-8", "ignore"))
                if any(marker in text.lower() for marker in chapter_markers):
                    return text
    except Exception as exc:
        log_error(args, "Reference evaluation read failed", exc, {"reference": reference_path})
    return ""


def evaluate_records(
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
) -> dict[str, object]:
    flag_counts: dict[str, int] = {}
    chunk_flags = []
    final_text = chapter_plain_text_from_records(records)
    graph = build_chapter_entity_graph(records, chapter_name)
    for record in records:
        current = record.final or record.draft
        flags = validate_record_translation(record, record.draft, current)
        for flag in flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        if flags:
            chunk_flags.append({"chunk": record.index, "flags": flags})

    reference_text = reference_chapter_text(args)
    report: dict[str, object] = {
        "chapter": chapter_name,
        "chunks": len(records),
        "wrong_register_vous": flag_counts.get("wrong_register_vous", 0),
        "unexpected_vous": flag_counts.get("unexpected_vous", 0),
        "missing_inline_markers": flag_counts.get("missing_inline_markers", 0),
        "english_heavy": flag_counts.get("english_heavy", 0),
        "ascii_dialogue_quotes": flag_counts.get("ascii_dialogue_quotes", 0),
        "length_ratio": flag_counts.get("length_ratio", 0),
        "negation_risk": flag_counts.get("negation_risk", 0),
        "profanity_softened": flag_counts.get("profanity_softened", 0),
        "dialogue_punctuation_lost": flag_counts.get("dialogue_punctuation_lost", 0),
        "question_punctuation_lost": flag_counts.get("question_punctuation_lost", 0),
        "tense_present_drift": flag_counts.get("tense_present_drift", 0),
        "address_name_lost": sum(
            count for flag, count in flag_counts.items()
            if flag.startswith("address_name_lost_")
        ),
        "scene_name_lost": sum(
            count for flag, count in flag_counts.items()
            if flag.startswith("scene_name_lost_")
        ),
        "glossary_missing": sum(
            count for flag, count in flag_counts.items()
            if flag.startswith("glossary_missing_")
        ),
        "glossary_forbidden": sum(
            count for flag, count in flag_counts.items()
            if flag.startswith("glossary_forbidden_")
        ),
        "guillemets": final_text.count("\u00ab") + final_text.count("\u00bb"),
        "ascii_quotes": final_text.count('"'),
        "characters": graph.characters,
        "dialogue_edges": graph.dialogue_edges,
        "ambiguous_dialogue_chunks": graph.ambiguous_dialogue_chunks,
        "flag_counts": flag_counts,
        "chunk_flags": chunk_flags,
    }
    if reference_text and getattr(args, "chrf_evaluation", False):
        report["chrf_like_vs_reference"] = chrf_like(final_text, reference_text)
    if getattr(args, "grammar_check", False) and final_text:
        try:
            from lib.grammar_check import check_french
            grammar_issues = check_french(
                final_text,
                url=getattr(args, "languagetool_url", "http://localhost:8081"),
                min_length=getattr(args, "grammar_min_chunk_length", 40),
                max_errors=getattr(args, "grammar_max_errors", 5),
            )
            if grammar_issues:
                report["grammar_issues"] = grammar_issues
                write_progress(
                    f"Grammar check ({chapter_name}): {len(grammar_issues)} issue(s) found."
                )
        except Exception:
            pass
    return report


def write_evaluation_report(
    args: argparse.Namespace,
    run_mode: RunMode,
    input_path: Path,
    output_path: Path,
    evaluations: list[dict[str, object]],
) -> None:
    if not evaluations:
        return
    report_dir = Path(args.evaluation_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    label = output_chapter_label(args.chapter, args.all_chapters, args.xhtml_index)
    report_path = report_dir / f"{safe_output_stem(run_mode.output_name)}-{label}-{safe_output_stem(input_path.stem)}.json"
    payload = {
        "book": str(input_path),
        "model": run_mode.output_name,
        "draft_model": run_mode.draft_model,
        "patcher": run_mode.patcher,
        "output_path": str(output_path),
        "reference": args.eval_reference or "",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "chapters": evaluations,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_progress(f"Evaluation report: {report_path}")
