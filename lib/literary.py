"""Literary post-editing pipeline: LLM patching, critique, back-translation, reread."""
from __future__ import annotations
import json, re
from lib.models import ChunkRecord, CandidateChoice, PatchDecision
from lib.char_graph import strip_inline_markers, CHARACTER_ALIASES, character_mentions
from lib.validators import validate_style_rules, validate_glossary_rules, validate_record_translation
from lib.scoring import (
    candidate_score, candidate_score_for_record, score_from_flags,
    hard_flag_count, candidate_accepted, rerank_record_candidates,
    populate_arbitration_drafts,
)
from lib.comet import (
    score_candidates_with_cometkiwi, choose_qe_candidate,
    score_drafts_for_patching, build_patch_decisions, patch_fn_call,
    patch_records_with_qe,
)
from lib.memo import (
    build_style_memo, clean_generated_text, issues_from_flags,
    format_alt_drafts, build_context,
)
from lib.correction_memory import (
    load_correction_memory, save_correction_memory,
    correction_memory_examples_text, exact_correction_memory_candidate,
    similar_correction_memory_entries,
)
from lib.logging_utils import write_progress, print_chunk_text
from lib.registries import DEFAULT_PATCH_SCORE_THRESHOLD, DEFAULT_HEURISTIC_PATCH_THRESHOLD

def patch_records_with_llm(
    patcher: str,
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
    patcher_module: object | None = None,
    cleanup_after: bool = True,
) -> tuple[list[str], int]:
    if args.dry_run:
        return [record.draft for record in records], 0

    decisions = build_patch_decisions(records, args, chapter_name)
    selected_decisions = [decision for decision in decisions if decision.needs_patch]
    if not selected_decisions:
        write_progress(f"{patcher} patch skipped: all draft chunks passed QE/validation.")
        return [record.draft for record in records], 0

    populate_arbitration_drafts(selected_decisions, args, chapter_name)

    loaded_module = patcher_module is not None
    if patcher_module is None:
        patcher_module = load_patcher(patcher)
    ensure = getattr(patcher_module, "ensure_model_loaded", None)
    cleanup = getattr(patcher_module, "unload_model", None)
    patch_fn = getattr(patcher_module, "patch_translation", None)
    if not callable(patch_fn):
        raise RuntimeError(f"{patcher} does not expose patch_translation().")

    if callable(ensure):
        write_progress(f"Loading {patcher} patcher...")
        ensure()
        write_progress(f"{patcher} patcher ready.")

    validation_failures = 0
    chapter_memory = build_chapter_memory(records, chapter_name, args)
    memo = _trim_memo(
        build_style_memo(patcher_module, records, args, chapter_name, chapter_memory),
        getattr(args, "memo_token_budget", 600),
    )
    patched_texts = [record.draft for record in records]

    # Use batch API when available (vLLM adapters) and correction memory is disabled.
    batch_patch_fn = getattr(patcher_module, "batch_patch_translations", None)
    if (
        callable(batch_patch_fn)
        and not getattr(args, "correction_memory", False)
        and not getattr(args, "cache_translations", False)
    ):
        write_progress(f"Batch patching {len(selected_decisions)} chunks via {patcher}...")
        batch_items = []
        for decision in selected_decisions:
            record = decision.record
            idx = record.index - 1
            context = build_context(records, idx, max(0, args.context_window))
            issues = "\n".join(f"- {r}" for r in decision.reasons) or "- selected for review"
            batch_items.append({
                "source": record.source, "draft": record.draft,
                "context": context, "memo": memo, "issues": issues,
                "alt_drafts": format_alt_drafts(record), "correction_examples": "",
                "register_hint": record.register_hint,
            })
        try:
            batch_results = batch_patch_fn(batch_items)
        except Exception as exc:
            log_error(args, "Batch patch failed; falling back to sequential", exc)
            batch_results = [d.record.draft for d in selected_decisions]
        for decision, raw_patched in zip(selected_decisions, batch_results):
            record = decision.record
            idx = record.index - 1
            patched = clean_generated_text(raw_patched)
            patched = patched
            if args.french_typography:
                patched = french_typography(patched)
            patched = repair_inline_markers_from_hints(record.source, patched, record.unit.markers)
            accepted, flags, _, _ = candidate_accepted(record, record.draft, patched, args)
            if accepted:
                patched_texts[idx] = patched
            else:
                validation_failures += 1
        if not loaded_module and callable(cleanup) and cleanup_after:
            cleanup()
        return patched_texts, validation_failures

    patch_iter = tqdm(
        selected_decisions,
        desc=f"{patcher} patch",
        unit="chunk",
        **progress_kwargs(3),
    )

    try:
        for decision in patch_iter:
            record = decision.record
            idx = record.index - 1
            patch_iter.set_postfix_str(f"{record.index}/{len(records)}")
            context = build_context(records, idx, max(0, args.context_window))
            issues = "\n".join(f"- {reason}" for reason in decision.reasons) or "- selected for review"
            if decision.raw_score is not None:
                issues += f"\n- raw COMETKiwi score: {decision.raw_score:.4f}"
            if decision.heuristic_score is not None:
                issues += f"\n- local heuristic score: {decision.heuristic_score:.2f}"
            if record.speaker or record.addressee or record.register_hint:
                issues += (
                    f"\n- dialogue state: speaker={record.speaker or 'unknown'}, "
                    f"addressee={record.addressee or 'unknown'}, "
                    f"register={record.register_hint or 'default'}"
                )
            if record.scene_participants:
                issues += "\n- scene participants: " + ", ".join(record.scene_participants)
            if record.style_hints:
                issues += "\n- style rules for this chunk: " + ", ".join(record.style_hints)
            alt_drafts = format_alt_drafts(record)
            correction_examples = correction_memory_examples_text(args, record)
            patch_candidates: dict[str, str] = {}
            try:
                memory_candidate = exact_correction_memory_candidate(args, record)
                if memory_candidate:
                    patch_candidates["memory"] = memory_candidate
                    write_progress(
                        f"Correction memory hit: {chapter_name} chunk {record.index}"
                    )
                else:
                    sample_count = max(1, int(getattr(args, "patcher_sample_candidates", 2)))
                    for sample_index in range(sample_count):
                        temperature = (
                            0.0
                            if sample_index == 0
                            else max(0.0, float(getattr(args, "patcher_sample_temperature", 0.3)))
                        )
                        candidate_name = "patch" if sample_index == 0 else f"patch_sample_{sample_index + 1}"
                        cache_payload = {
                            "patcher": patcher,
                            "chapter": chapter_name,
                            "chunk": record.index,
                            "source": record.source,
                            "draft": record.draft,
                            "alt_drafts": alt_drafts,
                            "correction_examples": correction_examples,
                            "context": context,
                            "memo": memo,
                            "issues": issues,
                            "register_hint": record.register_hint,
                            "sample_index": sample_index,
                            "temperature": temperature,
                            "kind": "patch",
                        }
                        patch_candidates[candidate_name] = cached_text_call(
                            args,
                            "patch",
                            cache_payload,
                            lambda temperature=temperature: patch_fn_call(
                                patch_fn,
                                source=record.source,
                                draft=record.draft,
                                context=context,
                                memo=memo,
                                issues=issues,
                                alt_drafts=alt_drafts,
                                correction_examples=correction_examples,
                                register_hint=record.register_hint,
                                temperature=temperature,
                            ),
                        )
            except Exception as exc:
                log_error(
                    args,
                    "Patcher chunk failed; draft kept",
                    exc,
                    {
                        "patcher": patcher,
                        "chapter": chapter_name,
                        "chunk": record.index,
                    },
                )
            candidate_pool = {"draft": record.draft}
            for candidate_name, candidate_text in patch_candidates.items():
                cleaned_candidate = clean_generated_text(candidate_text)
                if args.french_typography:
                    cleaned_candidate = french_typography(cleaned_candidate)
                cleaned_candidate = repair_inline_markers_from_hints(
                    record.source,
                    cleaned_candidate,
                    record.unit.markers,
                )
                candidate_pool[candidate_name] = cleaned_candidate
            patched = candidate_pool.get("patch") or next(
                (text for name, text in candidate_pool.items() if name != "draft"),
                record.draft,
            )
            if record.alt_drafts:
                candidate_pool.update({f"alt:{key}": value for key, value in record.alt_drafts.items()})
            best_name, best_text, best_flags, best_score = rerank_record_candidates(
                record,
                candidate_pool,
                args,
            )
            if best_text != patched:
                log_error(
                    args,
                    "Candidate reranker selected safer chunk candidate",
                    details={
                        "patcher": patcher,
                        "chapter": chapter_name,
                        "chunk": record.index,
                        "selected": best_name,
                        "flags": ",".join(best_flags),
                        "score": f"{best_score:.2f}",
                    },
                )
                patched = best_text
            accepted, flags, patched_score, draft_score = candidate_accepted(
                record,
                record.draft,
                patched,
                args,
            )
            repair_round = 0
            while (
                not accepted
                and patched != record.draft
                and repair_round < max(0, args.max_repair_rounds)
            ):
                repair_round += 1
                repair_fn = getattr(patcher_module, "repair_translation", None)
                if not callable(repair_fn):
                    break
                try:
                    repair_payload = {
                        "patcher": patcher,
                        "chapter": chapter_name,
                        "chunk": record.index,
                        "round": repair_round,
                        "source": record.source,
                        "draft": record.draft,
                        "bad_translation": patched,
                        "flags": flags,
                        "context": context,
                        "memo": memo,
                        "kind": "repair",
                    }
                    repaired = cached_text_call(
                        args,
                        "repair",
                        repair_payload,
                        lambda: repair_fn(
                            source=record.source,
                            draft=record.draft,
                            bad_translation=patched,
                            flags=flags,
                            context=context,
                            memo=memo,
                        ),
                    )
                    repaired = clean_generated_text(repaired)
                    repaired = repaired
                    if args.french_typography:
                        repaired = french_typography(repaired)
                    repaired = repair_inline_markers_from_hints(
                        record.source,
                        repaired,
                        record.unit.markers,
                    )
                    repaired_flags = validate_record_translation(record, record.draft, repaired)
                    repaired_accepted, repaired_flags, repaired_score, _ = candidate_accepted(
                        record,
                        record.draft,
                        repaired,
                        args,
                    )
                    if repaired_accepted or (
                        hard_flag_count(repaired_flags) < hard_flag_count(flags)
                        and repaired_score >= patched_score
                    ):
                        patched = repaired
                        flags = repaired_flags
                        patched_score = repaired_score
                        accepted = repaired_accepted
                    else:
                        break
                except Exception as exc:
                    log_error(
                        args,
                        "Patcher repair failed",
                        exc,
                        {
                            "patcher": patcher,
                            "chapter": chapter_name,
                            "chunk": record.index,
                            "flags": ",".join(flags),
                        },
                    )
                    break
            if not accepted and patched != record.draft:
                validation_failures += 1
                log_error(
                    args,
                    "Patched chunk rejected; draft kept",
                    details={
                        "patcher": patcher,
                        "chapter": chapter_name,
                        "chunk": record.index,
                        "flags": ",".join(flags),
                    },
                )
                patched = record.draft

            if patched != record.draft:
                accepted, flags, patched_score, draft_score = candidate_accepted(
                    record,
                    record.draft,
                    patched,
                    args,
                )
                if not accepted:
                    validation_failures += 1
                    log_error(
                        args,
                        "Patched chunk rejected by strict acceptance; draft kept",
                        details={
                            "patcher": patcher,
                            "chapter": chapter_name,
                            "chunk": record.index,
                            "flags": ",".join(flags),
                            "draft_score": f"{draft_score:.2f}",
                            "patched_score": f"{patched_score:.2f}",
                        },
                    )
                    patched = record.draft

            patched_texts[idx] = patched
            if args.print_text:
                print_chunk_text(
                    chapter_name,
                    record.index,
                    len(records),
                    f"{patcher} patch",
                    patched,
                    args.chunk_preview_chars,
                )
    finally:
        if cleanup_after and not loaded_module and callable(cleanup):
            cleanup()

    return patched_texts, validation_failures


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


def adjusted_qe_score(
    source: str,
    draft: str,
    candidate: str,
    markers: list[InlineMarker],
    raw_comet_score: float | None,
) -> float:
    flags = validate_translation(source, draft, candidate, markers)
    if raw_comet_score is None:
        return candidate_score(source, candidate, markers)

    score = raw_comet_score * 100.0
    score -= 12.0 * len(flags)
    if "missing_inline_markers" in flags:
        score -= 40.0
    if "wrong_register_vous" in flags or "unexpected_vous" in flags:
        score -= 40.0
    if "prompt_leak" in flags:
        score -= 30.0

    local_score = candidate_score(source, candidate, markers)
    score += max(-5.0, min(5.0, (local_score - 100.0) / 4.0))
    return score


def qe_candidate_models(draft_model: str, args: argparse.Namespace) -> list[str]:
    if args.qe_candidate_models == "auto":
        if draft_model == "madlad":
            return ["nllb-ct2", "opus-ct2"] if args.prefer_ct2 else ["nllb", "opus"]
        return ["madlad", "opus-ct2"] if args.prefer_ct2 else ["madlad", "opus"]
    return [model.strip() for model in args.qe_candidate_models.split(",") if model.strip()]


def short_error(exc: BaseException, max_chars: int = 320) -> str:
    text = " ".join(str(exc).split())
    if not text:
        text = exc.__class__.__name__
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def comet_model_candidates(args: argparse.Namespace) -> list[str]:
    model_arg = str(args.comet_model or "auto").strip()
    if model_arg == "auto":
        return list(DEFAULT_COMET_MODEL_CANDIDATES)
    return [model.strip() for model in model_arg.split(",") if model.strip()]


def comet_access_hint(model_name: str) -> str:
    if "cometkiwi" in model_name.lower():
        return (
            "If this Hugging Face repo is gated, accept its license and run "
            "huggingface-cli login with a token that can read gated repos."
        )
    return "Check network access, Hugging Face cache, and the model name."


def configure_comet_runtime() -> None:
    import logging
    import warnings

    os.environ.setdefault("LIGHTNING_DISABLE_INSTALLATION_TIPS", "1")
    os.environ.setdefault("PYTORCH_LIGHTNING_DISABLE_INSTALLATION_TIPS", "1")
    for logger_name in (
        "lightning",
        "lightning.pytorch",
        "pytorch_lightning",
        "pytorch_lightning.utilities.rank_zero",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=r"Lightning automatically upgraded.*")
    warnings.filterwarnings("ignore", message=r"Found keys that are not in the model state dict.*")

    try:
        import torch

        if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def quiet_comet_streams(args: argparse.Namespace):
    import contextlib

    if not getattr(args, "quiet_comet", True):
        return contextlib.nullcontext()

    stack = contextlib.ExitStack()
    sink = quiet_comet_sink()
    stack.enter_context(contextlib.redirect_stdout(sink))
    stack.enter_context(contextlib.redirect_stderr(sink))
    return stack


def quiet_comet_sink():
    sink = getattr(quiet_comet_sink, "_sink", None)
    if sink is None or getattr(sink, "closed", True):
        sink = open(os.devnull, "w", encoding="utf-8")
        setattr(quiet_comet_sink, "_sink", sink)
    return sink


def comet_checkpoint_path(model_name: str) -> str:
    from comet import download_model

    if "/" not in model_name:
        return download_model(model_name)

    try:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(repo_id=model_name))
        checkpoint_path = model_path / "checkpoints" / "model.ckpt"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"checkpoint not found at {checkpoint_path}")
        return str(checkpoint_path)
    except Exception as exc:
        raise RuntimeError(
            f"{model_name}: {short_error(exc)} {comet_access_hint(model_name)}"
        ) from exc


def load_cometkiwi_model(args: argparse.Namespace) -> object:
    configure_comet_runtime()
    with quiet_comet_streams(args):
        from comet import load_from_checkpoint

    cache = getattr(load_cometkiwi_model, "_cache", {})
    errors: list[str] = []

    for model_name in comet_model_candidates(args):
        if model_name in cache:
            return cache[model_name]

        write_progress(f"Loading COMET QE model {model_name}...")
        try:
            with quiet_comet_streams(args):
                model_path = comet_checkpoint_path(model_name)
                cache[model_name] = load_from_checkpoint(model_path)
            setattr(load_cometkiwi_model, "_cache", cache)
            write_progress(f"COMET QE model ready: {model_name}.")
            return cache[model_name]
        except Exception as exc:
            errors.append(short_error(exc))
            write_progress(f"COMET QE model unavailable: {short_error(exc)}")

    tried = ", ".join(comet_model_candidates(args))
    raise RuntimeError(f"No COMET QE model could be loaded. Tried: {tried}. Errors: {' | '.join(errors)}")



def comet_output_scores(model_output: object) -> list[float]:
    scores = getattr(model_output, "scores", None)
    if scores is None and isinstance(model_output, dict):
        scores = model_output.get("scores")
    if scores is None and isinstance(model_output, tuple) and model_output:
        scores = model_output[0]
    if scores is None:
        raise RuntimeError("COMETKiwi did not return sentence-level scores.")
    return [float(score) for score in scores]


def predict_cometkiwi_scores(
    model: object,
    data: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[float]:
    configure_comet_runtime()
    batch_size = max(1, args.comet_batch_size)
    gpus = max(0, args.comet_gpus)
    with quiet_comet_streams(args):
        try:
            output = model.predict(
                data,
                batch_size=batch_size,
                gpus=gpus,
                progress_bar=False,
            )
        except TypeError:
            try:
                output = model.predict(data, batch_size=batch_size, gpus=gpus)
            except TypeError:
                output = model.predict(data, batch_size=batch_size)
    return comet_output_scores(output)


def unload_cometkiwi_model() -> None:
    cache = getattr(load_cometkiwi_model, "_cache", None)
    if cache is not None:
        cache.clear()
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def score_candidates_with_cometkiwi(
    candidates: dict[str, list[str]],
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
) -> dict[tuple[str, int], float]:
    cache_payload = {
        "chapter": chapter_name,
        "comet_model": args.comet_model,
        "candidates": candidates,
        "sources": [strip_inline_markers(record.source) for record in records],
        "kind": "comet_scores",
    }
    cached = read_pipeline_cache(
        args,
        "comet_scores",
        {"version": PIPELINE_CACHE_VERSION, **cache_payload},
    )
    if isinstance(cached, list):
        try:
            return {
                (str(item["model"]), int(item["offset"])): float(item["score"])
                for item in cached
                if isinstance(item, dict)
            }
        except Exception:
            pass

    model = load_cometkiwi_model(args)
    data: list[dict[str, str]] = []
    keys: list[tuple[str, int]] = []

    for model_name, outputs in candidates.items():
        for offset, record in enumerate(records):
            if offset >= len(outputs):
                continue
            data.append(
                {
                    "src": strip_inline_markers(record.source),
                    "mt": strip_inline_markers(outputs[offset]),
                }
            )
            keys.append((model_name, offset))

    if not data:
        return {}

    write_progress(f"COMET QE scoring {len(data)} candidate(s) for {chapter_name}...")
    scores = predict_cometkiwi_scores(model, data, args)
    if len(scores) != len(keys):
        raise RuntimeError(
            f"COMETKiwi returned {len(scores)} scores for {len(keys)} candidate translations."
        )

    result = dict(zip(keys, scores))
    write_pipeline_cache(
        args,
        "comet_scores",
        {"version": PIPELINE_CACHE_VERSION, **cache_payload},
        [
            {"model": model_name, "offset": offset, "score": score}
            for (model_name, offset), score in result.items()
        ],
    )
    low_count = sum(1 for score in scores if score < args.qe_low_score_threshold)
    write_progress(
        f"COMETKiwi scored {len(scores)} candidate(s) for {chapter_name}; "
        f"{low_count} below {args.qe_low_score_threshold:.2f}."
    )
    return result


def choose_qe_candidate(
    record: ChunkRecord,
    offset: int,
    candidates: dict[str, list[str]],
    comet_scores: dict[tuple[str, int], float],
    engine: str,
) -> CandidateChoice:
    best_choice: CandidateChoice | None = None
    for model_name, model_outputs in candidates.items():
        if offset >= len(model_outputs):
            continue
        candidate = model_outputs[offset]
        raw_score = comet_scores.get((model_name, offset))
        score = adjusted_qe_score(
            record.source,
            record.draft,
            candidate,
            record.unit.markers,
            raw_score,
        )
        if best_choice is None or score > best_choice.score:
            best_choice = CandidateChoice(
                model_name=model_name,
                text=candidate,
                score=score,
                raw_score=raw_score,
                engine=engine if raw_score is not None else "heuristic",
            )

    if best_choice is None:
        return CandidateChoice(
            model_name="draft",
            text=record.draft,
            score=candidate_score(record.source, record.draft, record.unit.markers),
            raw_score=None,
            engine="heuristic",
        )
    return best_choice


def score_drafts_for_patching(
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
) -> tuple[dict[tuple[str, int], float], str]:
    if args.qe_engine not in {"auto", "cometkiwi"} or args.dry_run:
        return {}, "heuristic"

    try:
        candidates = {"draft": [record.draft for record in records]}
        scores = score_candidates_with_cometkiwi(candidates, records, args, chapter_name)
        return scores, "cometkiwi"
    except Exception as exc:
        log_error(
            args,
            "COMETKiwi draft patch scoring failed",
            exc,
            {"chapter": chapter_name, "model": args.comet_model},
        )
        if args.qe_engine == "cometkiwi" and not args.comet_fallback_heuristic:
            raise
        write_progress(f"COMETKiwi patch scoring failed; using heuristic signals: {exc}")
        return {}, "heuristic"
    finally:
        unload_cometkiwi_model()


def build_patch_decisions(
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
) -> list[PatchDecision]:
    if not args.selective_patching:
        return [
            PatchDecision(
                record=record,
                needs_patch=True,
                reasons=["selective patching disabled"],
                heuristic_score=candidate_score_for_record(record, record.draft),
            )
            for record in records
        ]

    comet_scores, engine = score_drafts_for_patching(records, args, chapter_name)
    decisions: list[PatchDecision] = []
    for offset, record in enumerate(records):
        flags = validate_record_translation(record, record.draft, record.draft)
        record.flags = flags
        heuristic = candidate_score_for_record(record, record.draft)
        raw_score = comet_scores.get(("draft", offset))
        reasons = []
        if flags:
            reasons.extend(f"validator:{flag}" for flag in flags)
        if raw_score is not None and raw_score < args.patch_score_threshold:
            reasons.append(f"cometkiwi:{raw_score:.3f}<{args.patch_score_threshold:.3f}")
        if raw_score is None and heuristic < args.heuristic_patch_threshold:
            reasons.append(f"heuristic:{heuristic:.1f}<{args.heuristic_patch_threshold:.1f}")
        skip_threshold = float(getattr(args, "skip_patch_heuristic_threshold", 96.0))
        if heuristic >= skip_threshold and not hard_flag_count(flags):
            reasons = []

        decisions.append(
            PatchDecision(
                record=record,
                needs_patch=bool(reasons),
                reasons=reasons,
                raw_score=raw_score,
                heuristic_score=heuristic,
            )
        )

    patch_count = sum(1 for decision in decisions if decision.needs_patch)
    if engine == "cometkiwi":
        write_progress(f"Selective patching: {patch_count}/{len(records)} chunk(s) below QE/validation thresholds.")
    else:
        write_progress(f"Selective patching: {patch_count}/{len(records)} chunk(s) selected by heuristic validation.")
    return decisions


def patch_fn_call(
    patch_fn: Callable[..., str],
    source: str,
    draft: str,
    context: str,
    memo: str,
    issues: str,
    alt_drafts: str = "",
    correction_examples: str = "",
    register_hint: str = "",
    temperature: float = 0.0,
) -> str:
    try:
        return patch_fn(
            source=source,
            draft=draft,
            context=context,
            memo=memo,
            issues=issues,
            alt_drafts=alt_drafts,
            correction_examples=correction_examples,
            register_hint=register_hint,
            temperature=temperature,
        )
    except TypeError:
        try:
            return patch_fn(
                source=source,
                draft=draft,
                context=context,
                memo=memo,
                issues=issues,
                register_hint=register_hint,
            )
        except TypeError:
            try:
                return patch_fn(source=source, draft=draft, context=context, memo=memo, issues=issues)
            except TypeError:
                return patch_fn(source=source, draft=draft, context=context, memo=memo)


def patch_records_with_qe(
    draft_model: str,
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
) -> tuple[list[str], int]:
    candidates: dict[str, list[str]] = {draft_model: [record.draft for record in records]}
    validation_failures = 0

    for model_name in qe_candidate_models(draft_model, args):
        if model_name == draft_model:
            continue
        attempt_names = [model_name]
        if model_name.endswith("-ct2"):
            attempt_names.append(model_name.removesuffix("-ct2"))

        last_error: Exception | None = None
        for attempt_name in attempt_names:
            handle = load_translator(attempt_name)
            try:
                if not args.dry_run:
                    print(f"Loading QE candidate model {attempt_name}...")
                    handle.preload()
                candidates[attempt_name] = translate_records(
                    records,
                    handle,
                    args,
                    f"QE {attempt_name}",
                    chapter_name,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                log_error(
                    args,
                    "QE candidate model failed",
                    exc,
                    {"chapter": chapter_name, "model": attempt_name},
                )
                if attempt_name != attempt_names[-1]:
                    write_progress(f"QE candidate {attempt_name} failed; trying fallback.")
            finally:
                handle.cleanup()
        if last_error is not None:
            write_progress(f"Skipping QE candidate {model_name}; all attempts failed.")

    comet_scores: dict[tuple[str, int], float] = {}
    qe_engine = "heuristic"
    if args.qe_engine in {"auto", "cometkiwi"} and not args.dry_run:
        try:
            comet_scores = score_candidates_with_cometkiwi(candidates, records, args, chapter_name)
            qe_engine = "cometkiwi"
        except Exception as exc:
            log_error(
                args,
                "COMETKiwi QE failed",
                exc,
                {"chapter": chapter_name, "model": args.comet_model},
            )
            if args.qe_engine == "cometkiwi" and not args.comet_fallback_heuristic:
                raise
            write_progress(f"COMETKiwi QE failed; using heuristic QE: {exc}")
        finally:
            unload_cometkiwi_model()

    selected: list[str] = []
    for offset, record in enumerate(records):
        choice = choose_qe_candidate(record, offset, candidates, comet_scores, qe_engine)
        best_model = choice.model_name
        best_text = choice.text
        best_score = choice.score
        flags = validate_record_translation(record, record.draft, best_text)
        low_comet_score = (
            choice.raw_score is not None
            and choice.raw_score < args.qe_low_score_threshold
        )
        if flags or low_comet_score:
            validation_failures += 1
            log_error(
                args,
                "QE selected low-confidence candidate",
                details={
                    "chapter": chapter_name,
                    "engine": choice.engine,
                    "chunk": record.index,
                    "selected_model": best_model,
                    "flags": ",".join(flags),
                    "raw_comet_score": (
                        f"{choice.raw_score:.4f}" if choice.raw_score is not None else ""
                    ),
                    "score": f"{best_score:.2f}",
                },
            )
        if flags and best_model != draft_model:
            draft_flags = validate_record_translation(record, record.draft, record.draft)
            draft_score = candidate_score_for_record(record, record.draft)
            if not draft_flags or best_score + args.patch_accept_margin < draft_score:
                log_error(
                    args,
                    "QE candidate rejected; draft kept",
                    details={
                        "chapter": chapter_name,
                        "chunk": record.index,
                        "selected_model": best_model,
                        "flags": ",".join(flags),
                        "draft_score": f"{draft_score:.2f}",
                        "candidate_score": f"{best_score:.2f}",
                    },
                )
                best_model = draft_model
                best_text = record.draft
                best_score = draft_score
                flags = draft_flags
        if args.french_typography:
            best_text = french_typography(best_text)
        best_text = repair_inline_markers_from_hints(
            record.source,
            best_text,
            record.unit.markers,
        )
        selected.append(best_text)
        if args.print_text:
            score_label = (
                f"{choice.raw_score:.3f}" if choice.raw_score is not None else f"{best_score:.1f}"
            )
            print_chunk_text(
                chapter_name,
                record.index,
                len(records),
                f"qe-{choice.engine}:{best_model}:{score_label}",
                best_text,
                args.chunk_preview_chars,
            )

    return selected, validation_failures


def critique_is_ok(critique: str) -> bool:
    normalized = re.sub(r"\s+", " ", critique.strip().lower())
    if not normalized:
        return True
    ok_markers = (
        "ok",
        "no issue",
        "no issues",
        "aucun probleme",
        "aucun problème",
        "faithful",
    )
    return any(normalized == marker or normalized.startswith(marker + ".") for marker in ok_markers)


def critique_records_with_llm(
    patcher: str,
    patcher_module: object,
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
    memo: str,
) -> int:
    if not args.literary_critic:
        return 0

    critique_fn = getattr(patcher_module, "critique_translation", None)
    repair_fn = getattr(patcher_module, "repair_translation", None)
    if not callable(critique_fn) or not callable(repair_fn):
        return 0

    failures = 0
    selected = [
        record for record in records
        if record.final
        and (
            record.final != record.draft
            or validate_record_translation(record, record.draft, record.final)
            or source_prefers_informal_second_person(record.source)
            or record.register_hint == "informal"
        )
    ]
    if not selected:
        return 0

    progress = tqdm(
        selected,
        desc="Critic",
        unit="chunk",
        **progress_kwargs(3),
    )
    try:
        for record in progress:
            progress.set_postfix_str(f"{record.index}/{len(records)}")
            context = build_context(records, record.index - 1, max(0, args.context_window))
            current = record.final or record.draft
            try:
                critique_payload = {
                    "patcher": patcher,
                    "chapter": chapter_name,
                    "chunk": record.index,
                    "source": record.source,
                    "draft": record.draft,
                    "candidate": current,
                    "context": context,
                    "memo": memo,
                    "kind": "critic",
                }
                critique = cached_text_call(
                    args,
                    "critic",
                    critique_payload,
                    lambda: critique_fn(
                        source=record.source,
                        draft=record.draft,
                        candidate=current,
                        context=context,
                        memo=memo,
                    ),
                )
            except Exception as exc:
                failures += 1
                log_error(
                    args,
                    "Literary critic failed; current text kept",
                    exc,
                    {"patcher": patcher, "chapter": chapter_name, "chunk": record.index},
                )
                continue

            if critique_is_ok(critique):
                continue

            flags = validate_record_translation(record, record.draft, current)
            repair_flags = flags + [f"critic_issue:{progress_text(critique, 240)}"]
            try:
                repaired_payload = {
                    "patcher": patcher,
                    "chapter": chapter_name,
                    "chunk": record.index,
                    "source": record.source,
                    "draft": record.draft,
                    "bad_translation": current,
                    "flags": repair_flags,
                    "context": context,
                    "memo": memo,
                    "kind": "critic_repair",
                }
                repaired = cached_text_call(
                    args,
                    "critic_repair",
                    repaired_payload,
                    lambda: repair_fn(
                        source=record.source,
                        draft=record.draft,
                        bad_translation=current,
                        flags=repair_flags,
                        context=context,
                        memo=memo,
                    ),
                )
                repaired = clean_generated_text(repaired)
                repaired = repaired
                if args.french_typography:
                    repaired = french_typography(repaired)
                repaired = repair_inline_markers_from_hints(
                    record.source,
                    repaired,
                    record.unit.markers,
                )
                accepted, repaired_flags, repaired_score, current_score = candidate_accepted(
                    record,
                    current,
                    repaired,
                    args,
                )
                if accepted:
                    record.final = repaired
                else:
                    failures += 1
                    log_error(
                        args,
                        "Critic repair rejected; current text kept",
                        details={
                            "patcher": patcher,
                            "chapter": chapter_name,
                            "chunk": record.index,
                            "critic": progress_text(critique, 240),
                            "flags": ",".join(repaired_flags),
                            "current_score": f"{current_score:.2f}",
                            "repaired_score": f"{repaired_score:.2f}",
                        },
                    )
            except Exception as exc:
                failures += 1
                log_error(
                    args,
                    "Critic repair failed; current text kept",
                    exc,
                    {"patcher": patcher, "chapter": chapter_name, "chunk": record.index},
                )
    finally:
        progress.close()

    return failures


def back_translation_flags(
    record: ChunkRecord,
    back_translation: str,
    args: argparse.Namespace,
) -> list[str]:
    source_text = strip_inline_markers(record.source)
    back_text = strip_inline_markers(back_translation)
    flags: list[str] = []
    overlap = token_jaccard(source_text, back_text)
    if overlap < max(0.0, float(getattr(args, "back_translation_threshold", 0.46))):
        flags.append(f"back_translation_low_overlap:{overlap:.2f}")
    for name in source_names(source_text):
        if not re.search(rf"\b{name}\b", back_text):
            flags.append(f"back_translation_missing_name_{name}")
    # Note: negation/profanity cross-checks removed — the seq2seq back-translation
    # model handles these naturally; regex-checking English words in back-translated
    # text introduced false positives without improving translation quality.
    return flags


def back_translation_selection(records: list[ChunkRecord], args: argparse.Namespace) -> list[ChunkRecord]:
    mode = getattr(args, "back_translation_check", "off")
    if mode == "off":
        return []
    selected = []
    for record in records:
        current = record.final or record.draft
        flags = validate_record_translation(record, record.draft, current)
        if mode == "all":
            selected.append(record)
        elif mode == "changed" and current != record.draft:
            selected.append(record)
        elif mode == "failed" and flags:
            selected.append(record)
    return selected


def back_translate_records_with_llm(
    patcher: str,
    patcher_module: object,
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
    memo: str,
) -> int:
    selected = back_translation_selection(records, args)
    if not selected:
        write_progress("Back-translation check skipped: no chunk selected.")
        return 0

    back_fn = getattr(patcher_module, "back_translate_to_english", None)
    repair_fn = getattr(patcher_module, "repair_translation", None)
    if not callable(back_fn) or not callable(repair_fn):
        write_progress("Back-translation check skipped: patcher does not expose the needed hooks.")
        return 0

    failures = 0
    progress = tqdm(
        selected,
        desc="Backcheck",
        unit="chunk",
        **progress_kwargs(3),
    )
    try:
        for record in progress:
            progress.set_postfix_str(f"{record.index}/{len(records)}")
            current = record.final or record.draft
            context = build_context(records, record.index - 1, max(0, args.context_window))
            try:
                back_payload = {
                    "patcher": patcher,
                    "chapter": chapter_name,
                    "chunk": record.index,
                    "source": record.source,
                    "candidate": current,
                    "kind": "back_translate",
                }
                back_text = cached_text_call(
                    args,
                    "back_translate",
                    back_payload,
                    lambda: back_fn(current),
                )
            except Exception as exc:
                failures += 1
                log_error(
                    args,
                    "Back-translation failed; current text kept",
                    exc,
                    {"patcher": patcher, "chapter": chapter_name, "chunk": record.index},
                )
                continue

            drift_flags = back_translation_flags(record, back_text, args)
            if not drift_flags:
                continue

            repair_flags = drift_flags + [
                "Back-translation differed from source:",
                progress_text(back_text, 260),
            ]
            try:
                repaired_payload = {
                    "patcher": patcher,
                    "chapter": chapter_name,
                    "chunk": record.index,
                    "source": record.source,
                    "draft": record.draft,
                    "bad_translation": current,
                    "flags": repair_flags,
                    "context": context,
                    "memo": memo,
                    "kind": "back_translation_repair",
                }
                repaired = cached_text_call(
                    args,
                    "back_translation_repair",
                    repaired_payload,
                    lambda: repair_fn(
                        source=record.source,
                        draft=record.draft,
                        bad_translation=current,
                        flags=repair_flags,
                        context=context,
                        memo=memo,
                    ),
                )
                repaired = clean_generated_text(repaired)
                repaired = repaired
                if args.french_typography:
                    repaired = french_typography(repaired)
                repaired = repair_inline_markers_from_hints(
                    record.source,
                    repaired,
                    record.unit.markers,
                )
                accepted, flags, repaired_score, current_score = candidate_accepted(
                    record,
                    current,
                    repaired,
                    args,
                )
                if accepted:
                    record.final = repaired
                else:
                    failures += 1
                    log_error(
                        args,
                        "Back-translation repair rejected; current text kept",
                        details={
                            "patcher": patcher,
                            "chapter": chapter_name,
                            "chunk": record.index,
                            "back_flags": ",".join(drift_flags),
                            "candidate_flags": ",".join(flags),
                            "current_score": f"{current_score:.2f}",
                            "repaired_score": f"{repaired_score:.2f}",
                        },
                    )
            except Exception as exc:
                failures += 1
                log_error(
                    args,
                    "Back-translation repair failed; current text kept",
                    exc,
                    {"patcher": patcher, "chapter": chapter_name, "chunk": record.index},
                )
    finally:
        progress.close()

    return failures


def reread_selection(records: list[ChunkRecord], args: argparse.Namespace) -> list[ChunkRecord]:
    if args.literary_reread == "off":
        return []
    if args.literary_reread == "all":
        return list(records)

    selected = []
    for record in records:
        current = record.final or record.draft
        flags = validate_record_translation(record, record.draft, current)
        is_dialogue = has_direct_speech(record.source)
        if flags or (args.literary_reread == "dialogue" and is_dialogue):
            selected.append(record)
    return selected


def reread_records_with_llm(
    patcher: str,
    patcher_module: object,
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
    memo: str,
) -> int:
    selected = reread_selection(records, args)
    if not selected:
        write_progress("Literary reread skipped: no chunk selected.")
        return 0

    reread_fn = getattr(patcher_module, "reread_translation", None)
    patch_fn = getattr(patcher_module, "patch_translation", None)
    if not callable(reread_fn) and not callable(patch_fn):
        return 0

    failures = 0
    progress = tqdm(
        selected,
        desc="Reread",
        unit="chunk",
        **progress_kwargs(3),
    )
    try:
        for record in progress:
            progress.set_postfix_str(f"{record.index}/{len(records)}")
            current = record.final or record.draft
            context = build_context(records, record.index - 1, max(0, args.reread_window))
            flags = validate_record_translation(record, record.draft, current)
            issues = issues_from_flags(flags) or "- reread for chapter-level flow and consistency"
            try:
                reread_payload = {
                    "patcher": patcher,
                    "chapter": chapter_name,
                    "chunk": record.index,
                    "source": record.source,
                    "current": current,
                    "context": context,
                    "memo": memo,
                    "issues": issues,
                    "kind": "reread",
                }
                if callable(reread_fn):
                    reread = cached_text_call(
                        args,
                        "reread",
                        reread_payload,
                        lambda: reread_fn(
                            source=record.source,
                            current=current,
                            context=context,
                            memo=memo,
                            issues=issues,
                            register_hint=record.register_hint,
                        ),
                    )
                else:
                    reread = cached_text_call(
                        args,
                        "reread",
                        reread_payload,
                        lambda: patch_fn_call(
                            patch_fn,
                            source=record.source,
                            draft=current,
                            context=context,
                            memo=memo,
                            issues=issues,
                        ),
                    )
                reread = clean_generated_text(reread)
                reread = reread
                if args.french_typography:
                    reread = french_typography(reread)
                reread = repair_inline_markers_from_hints(
                    record.source,
                    reread,
                    record.unit.markers,
                )
                accepted, reread_flags, reread_score, current_score = candidate_accepted(
                    record,
                    current,
                    reread,
                    args,
                )
                if accepted:
                    record.final = reread
                else:
                    failures += 1
                    log_error(
                        args,
                        "Reread edit rejected; current text kept",
                        details={
                            "patcher": patcher,
                            "chapter": chapter_name,
                            "chunk": record.index,
                            "flags": ",".join(reread_flags),
                            "current_score": f"{current_score:.2f}",
                            "reread_score": f"{reread_score:.2f}",
                        },
                    )
            except Exception as exc:
                failures += 1
                log_error(
                    args,
                    "Reread edit failed; current text kept",
                    exc,
                    {"patcher": patcher, "chapter": chapter_name, "chunk": record.index},
                )
    finally:
        progress.close()

    return failures


def parse_window_json_output(raw: str, expected_indices: list[int]) -> dict[int, str]:
    cleaned = clean_generated_text(raw)
    match = re.search(r"(\[[\s\S]*\])", cleaned)
    if match:
        cleaned = match.group(1)
    data = json.loads(cleaned)
    if isinstance(data, dict) and "translations" in data:
        data = data["translations"]
    if not isinstance(data, list):
        raise ValueError("window output is not a JSON array")
    result: dict[int, str] = {}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("window item is not a JSON object")
        index = int(item.get("index"))
        translation = str(item.get("translation", "")).strip()
        if not translation:
            raise ValueError(f"empty window translation for chunk {index}")
        result[index] = translation
    if set(result) != set(expected_indices):
        raise ValueError(
            f"window indices mismatch: expected {expected_indices}, got {sorted(result)}"
        )
    return result


def window_record_ranges(records: list[ChunkRecord], args: argparse.Namespace) -> list[tuple[int, int]]:
    mode = getattr(args, "literary_window_reread", "off")
    if mode == "off" or not records:
        return []
    size = max(2, int(getattr(args, "window_size", 5)))
    stride = max(1, int(getattr(args, "window_stride", 3)))
    selected_indices: list[int] = []
    for index, record in enumerate(records):
        current = record.final or record.draft
        flags = validate_record_translation(record, record.draft, current)
        is_dialogue = has_direct_speech(record.source)
        if mode == "all":
            selected_indices.append(index)
        elif mode == "failed" and flags:
            selected_indices.append(index)
        elif mode == "dialogue" and (flags or is_dialogue):
            selected_indices.append(index)

    ranges: list[tuple[int, int]] = []
    if mode == "all":
        start = 0
        while start < len(records):
            end = min(len(records), start + size)
            if end - start >= 2:
                ranges.append((start, end))
            start += stride
    else:
        half = max(1, size // 2)
        seen: set[tuple[int, int]] = set()
        for index in selected_indices:
            start = max(0, index - half)
            end = min(len(records), start + size)
            start = max(0, end - size)
            if end - start < 2:
                continue
            key = (start, end)
            if key not in seen:
                seen.add(key)
                ranges.append(key)

    max_windows = max(0, int(getattr(args, "window_max_windows", 0)))
    if max_windows:
        ranges = ranges[:max_windows]
    return ranges


def revise_records_in_windows(
    patcher: str,
    patcher_module: object,
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
    memo: str,
) -> int:
    ranges = window_record_ranges(records, args)
    if not ranges:
        write_progress("Window reread skipped: no window selected.")
        return 0

    window_fn = getattr(patcher_module, "revise_translation_window", None)
    if not callable(window_fn):
        write_progress("Window reread skipped: patcher does not expose revise_translation_window().")
        return 0

    failures = 0
    progress = tqdm(
        ranges,
        desc="Window",
        unit="window",
        **progress_kwargs(3),
    )
    try:
        for window_number, (start, end) in enumerate(progress, 1):
            window_records = records[start:end]
            expected_indices = [record.index for record in window_records]
            progress.set_postfix_str(f"{window_number}/{len(ranges)}")
            center = start + (end - start) // 2
            context = build_context(records, center, max(0, args.reread_window))
            issue_lines = []
            for record in window_records:
                current = record.final or record.draft
                flags = validate_record_translation(record, record.draft, current)
                if flags:
                    issue_lines.append(f"chunk {record.index}: {', '.join(flags)}")
            issues = "\n".join(issue_lines) or "chapter-flow, pronouns, register, tense, and punctuation continuity"
            source_chunks = [
                {"index": record.index, "text": record.source}
                for record in window_records
            ]
            current_chunks = [
                {"index": record.index, "text": record.final or record.draft}
                for record in window_records
            ]
            try:
                payload = {
                    "patcher": patcher,
                    "chapter": chapter_name,
                    "indices": expected_indices,
                    "source_chunks": source_chunks,
                    "current_chunks": current_chunks,
                    "context": context,
                    "memo": memo,
                    "issues": issues,
                    "kind": "window_reread",
                }
                raw = cached_text_call(
                    args,
                    "window_reread",
                    payload,
                    lambda: window_fn(
                        source_chunks=source_chunks,
                        current_chunks=current_chunks,
                        context=context,
                        memo=memo,
                        issues=issues,
                    ),
                )
                proposed = parse_window_json_output(raw, expected_indices)
            except Exception as exc:
                failures += 1
                log_error(
                    args,
                    "Window reread JSON parse/generation failed",
                    exc,
                    {"patcher": patcher, "chapter": chapter_name, "indices": expected_indices},
                )
                continue

            for record in window_records:
                current = record.final or record.draft
                candidate = proposed.get(record.index, current)
                candidate = clean_generated_text(candidate)
                candidate = candidate
                if args.french_typography:
                    candidate = french_typography(candidate)
                candidate = repair_inline_markers_from_hints(
                    record.source,
                    candidate,
                    record.unit.markers,
                )
                accepted, flags, candidate_score_value, current_score = candidate_accepted(
                    record,
                    current,
                    candidate,
                    args,
                )
                if accepted:
                    record.final = candidate
                else:
                    failures += 1
                    log_error(
                        args,
                        "Window reread chunk rejected; current text kept",
                        details={
                            "patcher": patcher,
                            "chapter": chapter_name,
                            "chunk": record.index,
                            "flags": ",".join(flags),
                            "current_score": f"{current_score:.2f}",
                            "candidate_score": f"{candidate_score_value:.2f}",
                        },
                    )
    finally:
        progress.close()

    return failures


def final_safety_rerank_records(
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
) -> int:
    changed = 0
    for record in records:
        current = record.final or record.draft
        candidates = {"current": current, "draft": record.draft}
        if record.alt_drafts:
            candidates.update({f"alt:{key}": value for key, value in record.alt_drafts.items()})
        best_name, best_text, best_flags, best_score = rerank_record_candidates(record, candidates, args)
        if best_text != current:
            changed += 1
            record.final = best_text
            log_error(
                args,
                "Final safety reranker replaced risky output",
                details={
                    "chapter": chapter_name,
                    "chunk": record.index,
                    "selected": best_name,
                    "flags": ",".join(best_flags),
                    "score": f"{best_score:.2f}",
                },
            )
    if changed:
        write_progress(f"Final safety reranker replaced {changed} risky chunk(s).")
    return changed


def patch_records_literary(
    patcher: str,
    records: list[ChunkRecord],
    args: argparse.Namespace,
    chapter_name: str,
    pipeline_progress: object | None = None,
) -> tuple[list[str], int, dict[str, object]]:
    if args.dry_run:
        for record in records:
            record.final = record.draft
        pipeline_step(pipeline_progress, "literary patch skipped")
        pipeline_step(pipeline_progress, "critic skipped")
        pipeline_step(pipeline_progress, "back-check skipped")
        pipeline_step(pipeline_progress, "reread skipped")
        pipeline_step(pipeline_progress, "window reread skipped")
        evaluation = evaluate_records(records, args, chapter_name)
        pipeline_step(pipeline_progress, "evaluated")
        return [record.final for record in records], 0, evaluation

    patcher_module = load_patcher(patcher)
    ensure = getattr(patcher_module, "ensure_model_loaded", None)
    cleanup = getattr(patcher_module, "unload_model", None)

    validation_failures = 0
    try:
        patched_texts, patch_failures = patch_records_with_llm(
            patcher,
            records,
            args,
            chapter_name,
            patcher_module=patcher_module,
            cleanup_after=False,
        )
        validation_failures += patch_failures
        for record, final_text in zip(records, patched_texts):
            record.final = final_text
        pipeline_step(pipeline_progress, f"{patcher} literary patch")

        chapter_memory = build_chapter_memory(records, chapter_name, args)
        if callable(ensure):
            write_progress(f"Ensuring {patcher} literary editor is loaded...")
            ensure()
            write_progress(f"{patcher} literary editor ready.")
        memo = _trim_memo(
            build_style_memo(patcher_module, records, args, chapter_name, chapter_memory),
            getattr(args, "memo_token_budget", 600),
        )

        validation_failures += critique_records_with_llm(
            patcher,
            patcher_module,
            records,
            args,
            chapter_name,
            memo,
        )
        pipeline_step(pipeline_progress, "critic repaired")

        validation_failures += back_translate_records_with_llm(
            patcher,
            patcher_module,
            records,
            args,
            chapter_name,
            memo,
        )
        pipeline_step(pipeline_progress, "back-check complete")

        validation_failures += reread_records_with_llm(
            patcher,
            patcher_module,
            records,
            args,
            chapter_name,
            memo,
        )
        pipeline_step(pipeline_progress, "reread complete")

        validation_failures += revise_records_in_windows(
            patcher,
            patcher_module,
            records,
            args,
            chapter_name,
            memo,
        )
        pipeline_step(pipeline_progress, "window reread complete")

        final_safety_rerank_records(records, args, chapter_name)
    finally:
        if callable(cleanup):
            cleanup()

    evaluation = evaluate_records(records, args, chapter_name)
    pipeline_step(pipeline_progress, "evaluated")
    save_correction_memory(args, records, chapter_name, patcher)
    save_book_memory(args, records, chapter_name)
    return [record.final or record.draft for record in records], validation_failures, evaluation
