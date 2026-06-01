"""COMET/CometKiwi quality-estimation and patch decision logic."""
from __future__ import annotations
import os, sys, re
from lib.models import ChunkRecord, PatchDecision, CandidateChoice
from lib.scoring import candidate_score_for_record
from lib.logging_utils import write_progress
from lib.registries import (
    DEFAULT_COMET_MODEL, DEFAULT_COMET_MODEL_CANDIDATES,
    DEFAULT_PATCH_SCORE_THRESHOLD, DEFAULT_HEURISTIC_PATCH_THRESHOLD,
)

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
