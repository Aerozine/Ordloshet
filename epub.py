#!/usr/bin/env python3
"""EPUB translator — main entry point and translation orchestration."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_ASCII", "1")
os.environ.setdefault("LIGHTNING_DISABLE_INSTALLATION_TIPS", "1")
os.environ.setdefault("PYTORCH_LIGHTNING_DISABLE_INSTALLATION_TIPS", "1")

try:
    from tqdm.std import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable: Iterable | None = None, **_: object) -> None:
            self.iterable = iterable if iterable is not None else []
        def __iter__(self) -> Iterable:
            return iter(self.iterable)
        def update(self, _: int = 1) -> None: return None
        def set_postfix_str(self, _: str) -> None: return None
        def close(self) -> None: return None
        @staticmethod
        def write(message: str, **_: object) -> None: print(message)

# --- lib imports ---
from lib.constants import PROMPT_LEAK_MARKERS, INLINE_TAGS, NAME_STOPWORDS, TRANSLATION_STOPWORDS
from lib.models import (
    CandidateChoice, ChapterEntityGraph, ChapterRunResult, ChunkRecord,
    InlineMarker, PatchDecision, RecordBatch, RunMode,
    TranslationBatchItem, TranslationUnit, TranslatorHandle,
)
from lib.char_graph import (
    CHARACTER_ALIASES, CHARACTER_GENDERS, CHARACTER_RELATIONSHIPS,
    FRENCH_DIALOGUE_MARKER_RE, FRENCH_PRESENT_DRIFT_RE,
    PLURAL_OR_FORMAL_SOURCE_RE, SECOND_PERSON_SOURCE_RE, SOURCE_PAST_RE, SPEECH_VERBS_RE,
    annotate_dialogue_state, build_chapter_entity_graph, canonical_character_name,
    character_mentions, character_pair_key, has_direct_speech,
    infer_source_addressee, infer_source_speaker, infer_unknown_dialogue_speaker,
    intimate_partner_of, intimate_pairs, likely_past_narration, likely_present_tense_drift,
    other_recent_participant, relationship_between, source_aliases,
    source_prefers_informal_second_person, strip_inline_markers,
    style_hints_for_record, text_without_quoted_dialogue, unique_keep_order,
)
from lib.registries import (
    COMPOSITE_MODELS, DEFAULT_BOOK_MEMORY_DIR, DEFAULT_CHUNK_PREVIEW_CHARS,
    DEFAULT_COMET_MODEL, DEFAULT_CORRECTION_MEMORY_DIR, DEFAULT_ERROR_LOG,
    DEFAULT_EVALUATION_DIR, DEFAULT_HEURISTIC_PATCH_THRESHOLD,
    DEFAULT_PATCH_SCORE_THRESHOLD, DEFAULT_SUMMARY_LOG, DEFAULT_TRANSLATION_CACHE_DIR,
    DEFAULT_BATCH_TOKEN_LIMIT, DIRECT_MODEL_MODULES, LITERARY_MODELS, PATCHER_MODULES,
    PIPELINE_CACHE_VERSION, PROGRESS_WIDTH, TRANSLATION_CACHE_VERSION,
)
from lib.text_utils import (
    extract_chapter_number, french_typography, log_text, maybe_ascii,
    normalize_text, number_to_words, progress_text, requested_chapter_number,
    safe_output_stem, sanitize_keyboard_text, should_translate, ui_text, word_tokens,
    words_to_number,
)
from lib.logging_utils import (
    append_summary, error_log_path, init_error_log, log_error, mark_logged,
    pipeline_step, print_chunk_text, print_error_log_location, progress_kwargs,
    progress_position, refresh_progress, set_pipeline_total, was_logged, write_progress,
)
from lib.epub_io import (
    extract_translation_units, first_heading_text, html_items_in_spine_order,
    is_non_story_item, is_numbered_chapter, item_name, list_xhtml_items,
    marker_nodes, normalize_marked_text, parse_html, replace_tag_text,
    score_chapter_candidate, select_chapter_items, write_epub_overwrite,
    marked_text_from_tag,
)
from lib.validators import (
    GLOSSARY_RULES, glossary_slug, repair_inline_markers_from_hints,
    source_has_glossary_term, source_names, text_matches_any,
    validate_glossary_rules, validate_record_translation, validate_style_rules,
    validate_translation, register_source_names, _register_chapter_names,
    marker_content_by_id,
)
from lib.batching import (
    estimate_token_count, make_record_batches, token_counts_for_records,
    translate_batch, _make_translation_batch_items, _collapse_translated_items,
    _short_chunk_groupable, _split_grouped_translation, _split_long_source,
    _pick_best_hypothesis,
)
from lib.translation_cache import (
    _append_chapter_checkpoint, _chapter_checkpoint_path, _clear_chapter_checkpoint,
    _read_chapter_checkpoint, cached_text_call, pipeline_cache_enabled, pipeline_cache_path,
    read_pipeline_cache, read_translation_cache, translation_cache_enabled,
    translation_cache_key, translation_cache_path, write_pipeline_cache,
    write_translation_cache, translation_cache_payload,
)
from lib.book_memory import (
    apply_book_register_memory, book_memory_path, book_memory_summary,
    load_book_memory, save_book_memory,
)
from lib.correction_memory import (
    content_token_set, correction_entry_key, correction_memory_examples_text,
    correction_memory_path, correction_memory_similarity, exact_correction_memory_candidate,
    load_correction_memory, normalize_memory_text, save_correction_memory,
    similar_correction_memory_entries, token_jaccard,
    _faiss_correction_memory_search,
)
from lib.memo import (
    _build_register_memo_line, _trim_memo, build_chapter_memory, build_context,
    build_style_memo, clean_generated_text, default_style_memo,
    extract_known_terms, extract_name_candidates, format_alt_drafts, issues_from_flags,
    _HALLUCINATION_RE,
)
from lib.scoring import (
    HARD_VALIDATION_FLAGS, _HARD_FLAG_PREFIXES, _HARD_VALIDATION_FLAGS_STATIC,
    candidate_accepted, candidate_score, candidate_score_for_record,
    hard_flag_count, populate_arbitration_drafts, rerank_record_candidates,
    score_from_flags,
)
from lib.comet import (
    adjusted_qe_score, build_patch_decisions, choose_qe_candidate,
    comet_access_hint, comet_checkpoint_path, comet_model_candidates,
    comet_output_scores, configure_comet_runtime, load_cometkiwi_model,
    patch_fn_call, predict_cometkiwi_scores, qe_candidate_models,
    quiet_comet_sink, quiet_comet_streams, score_candidates_with_cometkiwi,
    score_drafts_for_patching, short_error, unload_cometkiwi_model,
)
from lib.literary import (
    back_translate_records_with_llm, back_translation_flags, back_translation_selection,
    critique_is_ok, critique_records_with_llm, final_safety_rerank_records,
    parse_window_json_output, patch_records_literary, patch_records_with_llm,
    patch_records_with_qe, reread_records_with_llm, reread_selection,
    revise_records_in_windows, window_record_ranges,
)
from lib.evaluation import (
    chapter_plain_text_from_records, character_ngrams, chrf_like, evaluate_records,
    reference_chapter_text, strip_html_for_reference, write_evaluation_report,
)
from lib.args import parse_args

PROMPT_LEAK_MARKERS = (
    "English:",
    "French:",
    "Translate the following",
    "Return only",
    "Here is",
    "<|im_start|>",
    "<|im_end|>",
)

def configure_torch_runtime(args: argparse.Namespace) -> None:
    os.environ["ORDLOSHET_ATTN_IMPLEMENTATION"] = args.attn_implementation
    os.environ["ORDLOSHET_TORCH_COMPILE"] = "1" if args.torch_compile else "0"
    os.environ["ORDLOSHET_TORCH_COMPILE_MODE"] = args.torch_compile_mode
    os.environ["ORDLOSHET_TORCH_COMPILE_BACKEND"] = args.torch_compile_backend
    os.environ["ORDLOSHET_TORCH_COMPILE_STRICT"] = "1" if args.torch_compile_strict else "0"


def resolve_run_mode(args: argparse.Namespace) -> RunMode:
    if args.model in COMPOSITE_MODELS:
        draft_model, patcher = COMPOSITE_MODELS[args.model]
        literary = args.literary_mode or args.model in LITERARY_MODELS
        if literary and patcher == "qe":
            raise ValueError("Literary mode needs an LLM patcher, not --patcher qe.")
        return RunMode(
            output_name=args.model,
            draft_model=draft_model,
            patcher=patcher,
            direct_model=None,
            literary=literary,
        )

    if args.patcher != "none":
        draft_model = args.draft_model or args.model
        if draft_model not in {"madlad", "nllb", "nllb-ct2", "madlad-ct2", "alma-7b-r"}:
            raise ValueError(
                "--patcher requires --draft-model nllb|nllb-ct2|madlad|madlad-ct2|alma-7b-r "
                "or --model using one of those as draft."
            )
        if args.literary_mode and args.patcher == "qe":
            raise ValueError("Literary mode needs an LLM patcher, not --patcher qe.")
        return RunMode(
            output_name=f"{draft_model}-{args.patcher}",
            draft_model=draft_model,
            patcher=args.patcher,
            direct_model=None,
            literary=args.literary_mode,
        )

    return RunMode(
        output_name=args.model,
        draft_model=args.model,
        patcher="none",
        direct_model=args.model,
        literary=False,
    )


def load_translator(model: str) -> TranslatorHandle:
    module_name = DIRECT_MODEL_MODULES.get(model)
    if module_name is None:
        raise RuntimeError(f"Unknown translator model: {model}")

    module = importlib.import_module(module_name)
    translate_fn = (
        getattr(module, "large_translate", None)
        or getattr(module, "translate", None)
        or getattr(module, "largetranslate", None)
    )
    if translate_fn is None:
        raise RuntimeError(f"{module_name} does not expose a translation function.")

    cleanup_fn = getattr(module, "unload_model", None)
    if cleanup_fn is None:
        cleanup_fn = lambda: None
    preload_fn = getattr(module, "ensure_model_loaded", None)
    if preload_fn is None:
        preload_fn = lambda: None
    translate_many_fn = getattr(module, "translate_many", None)
    translate_many_nbest_fn = getattr(module, "translate_many_nbest", None)
    count_tokens_many_fn = getattr(module, "count_tokens_many", None)
    return TranslatorHandle(
        name=model,
        translate=translate_fn,
        cleanup=cleanup_fn,
        preload=preload_fn,
        translate_many=translate_many_fn if callable(translate_many_fn) else None,
        translate_many_nbest=translate_many_nbest_fn if callable(translate_many_nbest_fn) else None,
        count_tokens_many=count_tokens_many_fn if callable(count_tokens_many_fn) else None,
    )


def load_patcher(patcher: str):
    module_name = PATCHER_MODULES.get(patcher)
    if module_name is None:
        raise RuntimeError(f"Unknown patcher: {patcher}")
    return importlib.import_module(module_name)


def output_chapter_label(chapter: str, all_chapters: bool, xhtml_index: int | None) -> str:
    if all_chapters:
        return "All_Chapters"
    if xhtml_index is not None:
        return f"XHTML_{xhtml_index:02d}"

    chapter_number = requested_chapter_number(chapter)
    if chapter_number == 1:
        return "Chapter_One"
    if chapter_number is not None:
        word = number_to_words(chapter_number)
        return f"Chapter_{word.title().replace(' ', '_')}" if word else f"Chapter_{chapter_number}"
    return re.sub(r"[^A-Za-z0-9]+", "_", chapter).strip("_") or "Chapter"


def _apply_coref_resolution(records: list[ChunkRecord]) -> None:
    """Improve ambiguous speaker attribution using fastcoref coreference clusters.

    Each record's source text is processed independently; clusters containing a
    known character name are used to assign a speaker when the record is ambiguous.
    """
    try:
        from fastcoref import FCoref  # type: ignore[import]
    except ImportError:
        write_progress(
            "WARNING: fastcoref not installed; --coref-resolution has no effect. "
            "Run: pip install fastcoref"
        )
        return

    ambiguous_indices = {r.index for r in records if not r.speaker}
    if not ambiguous_indices:
        return

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    try:
        coref_model = FCoref(device=device)
        for record in records:
            if record.index not in ambiguous_indices:
                continue
            text = strip_inline_markers(record.source)
            preds = coref_model.predict(texts=[text])
            clusters = preds[0].get_clusters(as_strings=True)
            for cluster in clusters:
                character = None
                for mention in cluster:
                    clean = mention.strip().rstrip("'s")
                    resolved = CHARACTER_ALIASES.get(clean) or CHARACTER_ALIASES.get(clean.lower())
                    if resolved:
                        character = resolved
                        break
                if character:
                    for mention in cluster:
                        if mention.lower().strip() in {"he", "she", "they", "i", "me"}:
                            record.speaker = character
                            break
                if record.speaker:
                    break
    except Exception as exc:
        write_progress(f"WARNING: coref resolution failed: {exc}")


def chapter_entity_graph_summary(graph: ChapterEntityGraph, max_edges: int = 32) -> str:
    lines = [
        "Chapter entity graph:",
        "Characters: " + (", ".join(graph.characters) if graph.characters else "unknown"),
        "Aliases: "
        + ", ".join(
            f"{alias}={canonical}"
            for alias, canonical in sorted(graph.aliases.items())
            if alias != canonical
        ),
        "Relationships: "
        + ", ".join(
            f"{pair}={relationship}"
            for pair, relationship in sorted(graph.relationships.items())
        ),
        "Style rules:",
    ]
    lines.extend(f"- {rule}" for rule in graph.style_rules)
    if graph.dialogue_edges:
        lines.append("Dialogue edges:")
        for edge in graph.dialogue_edges[:max_edges]:
            lines.append(
                f"- chunk {edge['chunk']}: speaker={edge['speaker']}, "
                f"addressee={edge['addressee']}, register={edge['register']}"
            )
    if graph.ambiguous_dialogue_chunks:
        chunks = ", ".join(str(chunk) for chunk in graph.ambiguous_dialogue_chunks[:20])
        lines.append(f"Ambiguous dialogue chunks needing extra care: {chunks}")
    return "\n".join(lines)


def translate_records(
    records: list[ChunkRecord],
    handle: TranslatorHandle,
    args: argparse.Namespace,
    stage_label: str,
    chapter_name: str,
) -> list[str]:
    cached_outputs = read_translation_cache(records, handle, args)
    if cached_outputs is not None:
        write_progress(
            f"{stage_label}: using cached {handle.name} translations "
            f"for {chapter_name} ({len(cached_outputs)} chunk(s))."
        )
        if args.print_text:
            for record, translated in zip(records, cached_outputs):
                print_chunk_text(
                    chapter_name,
                    record.index,
                    len(records),
                    f"{stage_label.lower()} cached",
                    translated,
                    args.chunk_preview_chars,
                )
        return cached_outputs

    # Load checkpoint if enabled (partial progress from an interrupted run)
    checkpoint_path = _chapter_checkpoint_path(args, chapter_name, handle.name)
    checkpoint_data = _read_chapter_checkpoint(checkpoint_path)
    if checkpoint_data:
        write_progress(
            f"{stage_label}: resuming from checkpoint ({len(checkpoint_data)} chunk(s) already done)."
        )

    if not args.dry_run:
        write_progress(f"{stage_label}: cache miss for {handle.name}; loading model if needed.")
        handle.preload()

    outputs: list[str] = []
    total = len(records)
    progress = tqdm(
        total=total,
        desc=stage_label,
        unit="chunk",
        **progress_kwargs(3),
    )

    try:
        batches = make_record_batches(records, handle, args)
        for batch_number, record_batch in enumerate(batches, 1):
            batch = record_batch.records

            # Skip chunks already covered by the checkpoint
            remaining = [r for r in batch if r.index not in checkpoint_data]
            if len(remaining) < len(batch):
                for record in batch:
                    if record.index in checkpoint_data:
                        outputs.append(checkpoint_data[record.index])
                        progress.update(1)
                if not remaining:
                    continue
                batch = remaining

            # Auto-split long chunks and group very short adjacent chunks. Both
            # paths are collapsed back to one output per original record below.
            items = _make_translation_batch_items(batch, args)
            flat_texts = [part for item in items for part in item.parts]
            flat_records: list[ChunkRecord] = []
            for item in items:
                flat_records.extend([item.records[0]] * len(item.parts))
            progress.set_postfix_str(
                f"batch {batch_number}/{len(batches)}, tokens {record_batch.token_count}"
            )
            try:
                flat_translated = translate_batch(handle, flat_texts, args, records=flat_records)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if not args.keep_source_on_error:
                    log_error(
                        args,
                        f"{stage_label} batch failed",
                        exc,
                        {
                            "model": handle.name,
                            "chapter": chapter_name,
                            "batch_start": batch[0].index if batch else 0,
                            "batch_size": len(batch),
                            "batch_tokens": record_batch.token_count,
                        },
                    )
                    raise
                write_progress(
                    f"{stage_label} batch starting at chunk {batch[0].index} "
                    f"failed in {chapter_name}: {exc}"
                )
                translated_batch = [record.source for record in batch]
            else:
                if len(flat_translated) != len(flat_texts):
                    raise RuntimeError(
                        f"{handle.name} returned {len(flat_translated)} translations "
                        f"for {len(flat_texts)} prepared input(s)."
                    )
                translated_batch = _collapse_translated_items(
                    items,
                    flat_translated,
                    handle,
                    args,
                )

            if len(translated_batch) != len(batch):
                raise RuntimeError(
                    f"{handle.name} returned {len(translated_batch)} translations for {len(batch)} inputs."
                )

            for record, translated in zip(batch, translated_batch):
                if not translated or translated.strip() == "Model already loaded":
                    raise RuntimeError(f"{handle.name} returned no translated text.")
                if args.french_typography:
                    translated = french_typography(translated)
                translated = repair_inline_markers_from_hints(
                    record.source,
                    translated,
                    record.unit.markers,
                )
                translated = _HALLUCINATION_RE.sub("", translated).strip() or translated
                flags = validate_record_translation(record, record.source, translated)
                if "missing_inline_markers" in flags:
                    log_error(
                        args,
                        "Inline markers missing after translation",
                        details={
                            "model": handle.name,
                            "chapter": chapter_name,
                            "chunk": record.index,
                            "flags": ",".join(flags),
                        },
                    )
                outputs.append(translated)
                _append_chapter_checkpoint(checkpoint_path, record.index, translated)
                if args.print_text:
                    print_chunk_text(
                        chapter_name,
                        record.index,
                        total,
                        stage_label.lower(),
                        translated,
                        args.chunk_preview_chars,
                    )
            progress.update(len(batch))
    finally:
        progress.close()
    write_translation_cache(records, handle, args, outputs)
    _clear_chapter_checkpoint(checkpoint_path)
    return outputs


def translate_chapter(
    item: object,
    translator: TranslatorHandle,
    args: argparse.Namespace,
    pipeline_progress: object | None = None,
) -> int:
    soup = parse_html(item.get_content())
    units = extract_translation_units(soup)
    chapter_name = item_name(item)
    display_chapter_name = ui_text(chapter_name)
    total_units = len(units)
    write_progress(f"Chapter: {display_chapter_name} ({total_units} chunk(s))")

    records: list[ChunkRecord] = []
    for index, unit in enumerate(units, 1):
        source_text = unit.marked_text
        records.append(ChunkRecord(index=index, unit=unit, source=source_text))
        if args.print_source_text:
            print_chunk_text(
                display_chapter_name,
                index,
                total_units,
                "source",
                source_text,
                args.chunk_preview_chars,
            )
    build_chapter_entity_graph(records, display_chapter_name)
    apply_book_register_memory(records, args)
    _register_chapter_names(records, args)

    pipeline_step(pipeline_progress, f"extracted {total_units} chunk(s)")
    translated_texts = translate_records(records, translator, args, "Chunks", display_chapter_name)
    pipeline_step(pipeline_progress, "translated chunks")
    for record, translated in zip(records, translated_texts):
        replace_tag_text(record.unit.tag, translated, record.unit.markers)

    if not args.dry_run:
        item.set_content(soup.encode("utf-8"))

    return len(units)


def translate_chapter_patched(
    item: object,
    draft_translator: TranslatorHandle,
    run_mode: RunMode,
    args: argparse.Namespace,
    pipeline_progress: object | None = None,
) -> ChapterRunResult:
    soup = parse_html(item.get_content())
    units = extract_translation_units(soup)
    chapter_name = ui_text(item_name(item))
    total_units = len(units)
    write_progress(f"Chapter: {chapter_name} ({total_units} chunk(s))")

    records = [
        ChunkRecord(
            index=index,
            unit=unit,
            source=unit.marked_text,
        )
        for index, unit in enumerate(units, 1)
    ]
    build_chapter_entity_graph(records, chapter_name)
    apply_book_register_memory(records, args)

    # Register proper names for source_names() so validation is book-agnostic.
    _register_chapter_names(records, args)

    # Improve ambiguous speaker attribution with coreference resolution
    if getattr(args, "coref_resolution", False):
        _apply_coref_resolution(records)

    pipeline_step(pipeline_progress, f"extracted {total_units} chunk(s)")
    draft_texts = translate_records(records, draft_translator, args, "Draft", chapter_name)
    for record, draft in zip(records, draft_texts):
        record.draft = draft
    pipeline_step(pipeline_progress, "draft translated")

    if not getattr(args, "fused_models", False):
        draft_translator.cleanup()
    evaluation: dict[str, object] = {}
    if run_mode.literary:
        final_texts, validation_failures, evaluation = patch_records_literary(
            run_mode.patcher,
            records,
            args,
            chapter_name,
            pipeline_progress,
        )
    elif run_mode.patcher == "qe":
        final_texts, validation_failures = patch_records_with_qe(
            run_mode.draft_model,
            records,
            args,
            chapter_name,
        )
        pipeline_step(pipeline_progress, "qe selected")
    else:
        final_texts, validation_failures = patch_records_with_llm(
            run_mode.patcher,
            records,
            args,
            chapter_name,
        )
        pipeline_step(pipeline_progress, f"{run_mode.patcher} patched")
        for record, final_text in zip(records, final_texts):
            record.final = final_text
        evaluation = evaluate_records(records, args, chapter_name)

    for record, final_text in zip(records, final_texts):
        record.final = final_text
        replace_tag_text(record.unit.tag, final_text, record.unit.markers)
    if not evaluation:
        evaluation = evaluate_records(records, args, chapter_name)
    pipeline_step(pipeline_progress, "chapter applied")

    # Update entity memory with translations from this chapter
    if getattr(args, "entity_memory", False) and not args.dry_run:
        book_stem = getattr(args, "_current_book_stem", "")
        if book_stem:
            try:
                from lib.entity_memory import EntityMemory
                em = EntityMemory(book_stem)
                source_chunks = [r.source for r in records]
                translated_chunks = [r.final or r.draft for r in records]
                entities = em.extract_entities(" ".join(source_chunks))
                em.update_from_chapter(entities, source_chunks, translated_chunks)
                for warning in em.consistency_warnings(entities)[:5]:
                    write_progress(f"Entity: {warning}")
            except Exception as exc:
                write_progress(f"Entity memory update failed: {exc}")

    if not args.dry_run:
        item.set_content(soup.encode("utf-8"))

    return ChapterRunResult(
        chunks=len(units),
        validation_failures=validation_failures,
        evaluation=evaluation,
    )


def process_book(
    input_path: Path,
    args: argparse.Namespace,
    translator: TranslatorHandle,
    run_mode: RunMode,
) -> int:
    from ebooklib import epub

    if not input_path.exists():
        message = f"File not found: {input_path}"
        print(message)
        log_error(
            args,
            message,
            details={"model": args.model, "strategy": args.strategy, "book": input_path},
        )
        return 1

    book = epub.read_epub(str(input_path))
    setattr(args, "_current_book_stem", input_path.stem)
    print(f"Book: {ui_text(input_path.name)}")
    if args.list_xhtml:
        list_xhtml_items(book, epub)
        return 0

    try:
        chapter_items = select_chapter_items(
            book,
            epub,
            args.chapter,
            args.all_chapters,
            args.xhtml_index,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        log_error(
            args,
            "Chapter selection failed",
            exc,
            {"model": args.model, "strategy": args.strategy, "book": input_path},
        )
        return 1

    selected_names = ", ".join(ui_text(item_name(item)) for item in chapter_items)
    print(f"Selected {len(chapter_items)} chapter(s): {selected_names}")

    total_chunks = 0
    pipeline_total = len(chapter_items) * 2 + (0 if args.dry_run else 1)
    pipeline_progress = tqdm(
        total=pipeline_total,
        desc="Pipeline",
        unit="step",
        **progress_kwargs(1),
    )
    chapter_iter = tqdm(
        chapter_items,
        desc="Chapters",
        unit="chapter",
        **progress_kwargs(2),
    )
    try:
        for chapter_index, item in enumerate(chapter_iter, 1):
            chapter_iter.set_postfix_str(f"{chapter_index}/{len(chapter_items)}")
            pipeline_progress.set_postfix_str(f"chapter {chapter_index}/{len(chapter_items)}")
            total_chunks += translate_chapter(item, translator, args, pipeline_progress)
    finally:
        chapter_iter.close()

    if args.dry_run:
        pipeline_progress.close()
        print(f"Dry run complete: {total_chunks} chunk(s) found; no EPUB written.")
        return 0

    output_dir = Path("output") / run_mode.output_name.replace("-", "_") / args.strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    label = output_chapter_label(args.chapter, args.all_chapters, args.xhtml_index)
    output_path = output_dir / f"translated-{label}_{safe_output_stem(input_path.stem)}.epub"
    pipeline_progress.set_postfix_str("write epub")
    write_epub_overwrite(epub, output_path, book)
    pipeline_step(pipeline_progress, "wrote epub")
    pipeline_progress.close()
    print(f"\nSaved: {output_path}")
    return 0


def process_book_patched(
    input_path: Path,
    args: argparse.Namespace,
    draft_translator: TranslatorHandle,
    run_mode: RunMode,
    timings: dict[str, float],
) -> int:
    from ebooklib import epub

    if not input_path.exists():
        message = f"File not found: {input_path}"
        print(message)
        log_error(
            args,
            message,
            details={"model": args.model, "strategy": args.strategy, "book": input_path},
        )
        return 1

    book = epub.read_epub(str(input_path))
    setattr(args, "_current_book_stem", input_path.stem)
    print(f"Book: {ui_text(input_path.name)}")
    if args.list_xhtml:
        list_xhtml_items(book, epub)
        return 0

    try:
        chapter_items = select_chapter_items(
            book,
            epub,
            args.chapter,
            args.all_chapters,
            args.xhtml_index,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        log_error(
            args,
            "Chapter selection failed",
            exc,
            {"model": args.model, "strategy": args.strategy, "book": input_path},
        )
        return 1

    selected_names = ", ".join(ui_text(item_name(item)) for item in chapter_items)
    print(f"Selected {len(chapter_items)} chapter(s): {selected_names}")

    total_chunks = 0
    validation_failures = 0
    evaluations: list[dict[str, object]] = []
    per_chapter_steps = 9 if run_mode.literary else 4
    pipeline_total = len(chapter_items) * per_chapter_steps + (0 if args.dry_run else 1)
    pipeline_progress = tqdm(
        total=pipeline_total,
        desc="Pipeline",
        unit="step",
        **progress_kwargs(1),
    )
    chapter_iter = tqdm(
        chapter_items,
        desc="Chapters",
        unit="chapter",
        **progress_kwargs(2),
    )
    # Preload patcher model in a background thread while draft translation runs
    # (only when keeping both models loaded simultaneously)
    import threading as _threading
    _patcher_preload_thread: _threading.Thread | None = None
    if (
        getattr(args, "fused_models", False)
        and run_mode.patcher not in {"none", "qe"}
        and not args.dry_run
    ):
        def _preload_patcher() -> None:
            try:
                pm = load_patcher(run_mode.patcher)
                ensure_fn = getattr(pm, "ensure_model_loaded", None)
                if callable(ensure_fn):
                    ensure_fn()
            except Exception:
                pass
        _patcher_preload_thread = _threading.Thread(
            target=_preload_patcher, daemon=True, name="patcher-preload"
        )
        _patcher_preload_thread.start()
        write_progress(f"Async patcher preload: {run_mode.patcher} loading in background.")

    patch_start = time.perf_counter()
    try:
        for chapter_index, item in enumerate(chapter_iter, 1):
            chapter_iter.set_postfix_str(f"{chapter_index}/{len(chapter_items)}")
            pipeline_progress.set_postfix_str(f"chapter {chapter_index}/{len(chapter_items)}")

            # Auto-glossary: extract terms from chapter 1 source text before translating
            if (
                chapter_index == 1
                and getattr(args, "auto_glossary", False)
                and run_mode.patcher not in {"none", "qe"}
                and not args.dry_run
            ):
                try:
                    from lib.glossary_extractor import extract_glossary, merge_into_glossary
                    import lib.validators as _val
                    from bs4 import BeautifulSoup as _BS4
                    _soup = _BS4(item.get_content(), "lxml")
                    _source_text = _soup.get_text(" ")
                    _patcher_mod = load_patcher(run_mode.patcher)
                    _extracted = extract_glossary(_source_text, _patcher_mod)
                    if _extracted:
                        import sys as _sys
                        _epub_mod = _sys.modules[__name__]
                        _val.GLOSSARY_RULES = merge_into_glossary(  # type: ignore[attr-defined]
                            getattr(_val, "GLOSSARY_RULES", {}), _extracted
                        )
                        write_progress(
                            f"Auto-glossary: extracted {len(_extracted)} term(s) from chapter 1."
                        )
                except Exception as exc:
                    write_progress(f"Auto-glossary extraction failed: {exc}")

            # Wait for async patcher preload before first patching call
            if _patcher_preload_thread is not None and _patcher_preload_thread.is_alive():
                _patcher_preload_thread.join()
                _patcher_preload_thread = None

            chapter_result = translate_chapter_patched(
                item,
                draft_translator,
                run_mode,
                args,
                pipeline_progress,
            )
            total_chunks += chapter_result.chunks
            validation_failures += chapter_result.validation_failures
            if chapter_result.evaluation:
                evaluations.append(chapter_result.evaluation)
    finally:
        chapter_iter.close()
        if _patcher_preload_thread is not None:
            _patcher_preload_thread.join(timeout=30)
    timings["translate_and_patch_seconds"] = time.perf_counter() - patch_start

    if args.dry_run:
        pipeline_progress.close()
        print(f"Dry run complete: {total_chunks} chunk(s) found; no EPUB written.")
        return 0

    output_dir = Path("output") / run_mode.output_name.replace("-", "_") / args.strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    label = output_chapter_label(args.chapter, args.all_chapters, args.xhtml_index)
    output_path = output_dir / f"translated-{label}_{safe_output_stem(input_path.stem)}.epub"
    pipeline_progress.set_postfix_str("write epub")
    write_epub_overwrite(epub, output_path, book)
    write_evaluation_report(args, run_mode, input_path, output_path, evaluations)
    pipeline_step(pipeline_progress, "wrote epub")
    pipeline_progress.close()
    print(f"\nSaved: {output_path}")

    append_summary(
        args,
        {
            "book": str(input_path),
            "chunks": total_chunks,
            "draft_model": run_mode.draft_model,
            "model": run_mode.output_name,
            "output_path": str(output_path),
            "patcher": run_mode.patcher,
            "strategy": args.strategy,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "validation_failures": validation_failures,
            **{key: round(value, 3) for key, value in timings.items()},
        },
    )
    return 0


def _apply_book_profile(book_path: Path, args: argparse.Namespace) -> None:
    """Run auto-analysis on *book_path* and apply the resulting profile."""
    try:
        from lib.book_analyzer import analyze_book
        from lib.book_config import apply_book_config_dict
        profile = analyze_book(book_path)
        if profile:
            apply_book_config_dict(profile)
            # Surface detected style notes for LLM patchers
            notes = profile.get("style_notes", "")
            if notes:
                setattr(args, "_auto_style_notes", notes)
    except Exception as exc:
        print(f"WARNING: auto-analysis failed for {book_path.name}: {exc}")


def main() -> int:
    args = parse_args()

    if getattr(args, "book_config", ""):
        try:
            from lib.book_config import apply_book_config
            apply_book_config(args.book_config)
        except Exception as exc:
            print(f"WARNING: Could not load book config {args.book_config}: {exc}")

    validate_glossary_rules._enabled = bool(args.glossary_enforcement)
    setattr(validate_record_translation, "_french_nlp_enabled", bool(getattr(args, "french_nlp", False)))
    configure_torch_runtime(args)
    init_error_log(args)
    input_paths = [Path(book_path) for book_path in args.book]
    try:
        run_mode = resolve_run_mode(args)
    except Exception as exc:
        log_error(args, "Invalid run mode", exc)
        print(f"ERROR: {exc}")
        print_error_log_location(args)
        return 1

    print(f"\nModel: {run_mode.output_name.upper()}")
    if run_mode.patcher != "none":
        print(f"Draft model: {run_mode.draft_model.upper()}, patcher: {run_mode.patcher.upper()}")
    if run_mode.literary:
        print("Literary pipeline: enabled (critic, register validation, reread, evaluation).")
    print_error_log_location(args)
    print("Translation text mode: preserving Unicode accents.")
    translator: TranslatorHandle | None = None

    exit_code = 0
    try:
        timings: dict[str, float] = {}
        if args.dry_run or args.list_xhtml:
            print("Dry run: model weights will not be loaded.")
            translator = TranslatorHandle(
                name=run_mode.draft_model,
                translate=lambda text: text,
                cleanup=lambda: None,
                preload=lambda: None,
                translate_many=lambda texts, _: texts,
                count_tokens_many=None,
            )
        else:
            translator = load_translator(run_mode.draft_model)
            if translation_cache_enabled(args) and not args.refresh_translation_cache:
                print("Draft model load deferred until translation cache miss.")
            else:
                print("Loading draft model...")
                load_start = time.perf_counter()
                translator.preload()
                timings["draft_load_seconds"] = time.perf_counter() - load_start
                print("Model ready. Starting translation progress.")

        book_iter = tqdm(
            input_paths,
            desc="Books",
            unit="book",
            **progress_kwargs(0),
        )
        for book_index, input_path in enumerate(book_iter, 1):
            book_iter.set_postfix_str(f"{book_index}/{len(input_paths)}")

            # Auto-analyse the book when no manual config was supplied.
            if not getattr(args, "book_config", ""):
                _apply_book_profile(input_path, args)

            book_start = time.perf_counter()
            if run_mode.patcher == "none":
                result = process_book(input_path, args, translator, run_mode)
                timings["translate_seconds"] = time.perf_counter() - book_start
                append_summary(
                    args,
                    {
                        "book": str(input_path),
                        "draft_model": run_mode.draft_model,
                        "model": run_mode.output_name,
                        "patcher": run_mode.patcher,
                        "strategy": args.strategy,
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                        **{key: round(value, 3) for key, value in timings.items()},
                    },
                )
            else:
                result = process_book_patched(input_path, args, translator, run_mode, timings)
            if result != 0:
                exit_code = result
    except Exception as exc:
        if not was_logged(exc):
            log_error(args, "Unhandled run failure", exc)
        print(f"ERROR: {exc}")
        print_error_log_location(args)
        exit_code = 1
    finally:
        if translator is not None:
            translator.cleanup()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
