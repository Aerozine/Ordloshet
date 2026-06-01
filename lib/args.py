"""CLI argument parser for the EPUB translation pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path

from lib.registries import (
    COMPOSITE_MODELS,
    DEFAULT_BOOK_MEMORY_DIR,
    DEFAULT_COMET_MODEL,
    DEFAULT_CORRECTION_MEMORY_DIR,
    DEFAULT_ERROR_LOG,
    DEFAULT_EVALUATION_DIR,
    DEFAULT_HEURISTIC_PATCH_THRESHOLD,
    DEFAULT_PATCH_SCORE_THRESHOLD,
    DEFAULT_SUMMARY_LOG,
    DIRECT_MODEL_MODULES,
    LITERARY_MODELS,
    PATCHER_MODULES,
    PROGRESS_WIDTH,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate an EPUB to French.  Positional EPUB paths are equivalent to --book.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Positional EPUB files — allows: python epub.py mybook.epub
    parser.add_argument(
        "epub_files",
        nargs="*",
        metavar="EPUB",
        help="Input EPUB file(s). Equivalent to --book.",
    )
    parser.add_argument(
        "--model",
        default="madlad-ct2",
        choices=sorted(set(DIRECT_MODEL_MODULES) | set(COMPOSITE_MODELS)),
        help="Translation model (default: madlad-ct2).",
    )
    parser.add_argument(
        "--strategy",
        default="heavy",
        choices=["light", "heavy"],
        help="Context window strategy (default: heavy).",
    )
    parser.add_argument(
        "--draft-model",
        choices=sorted(DIRECT_MODEL_MODULES),
        help="Draft model for patch pipelines. Composite --model values set this automatically.",
    )
    parser.add_argument(
        "--patcher",
        default="none",
        choices=["none", "qe"] + sorted(PATCHER_MODULES),
        help="Optional patcher for draft translations.",
    )
    parser.add_argument(
        "--literary-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the document-aware literary pipeline: stronger register/format "
            "validators, critic repair, and reread pass. Enabled automatically by "
            "--model madlad-literary."
        ),
    )
    parser.add_argument(
        "--book",
        action="append",
        default=None,
        help="Input EPUB path. Repeat --book to process several books.",
    )
    parser.add_argument(
        "--chapter",
        default="1",
        help="Numbered chapter to translate. Default: 1 means the real Chapter One.",
    )
    parser.add_argument(
        "--xhtml-index",
        type=int,
        help="Translate the Nth XHTML spine item, 1-based. Use --list-xhtml to inspect positions.",
    )
    parser.add_argument(
        "--list-xhtml",
        action="store_true",
        help="List XHTML spine items and exit without loading a model.",
    )
    parser.add_argument(
        "--all-chapters",
        action="store_true",
        help="Translate every numbered chapter instead of the selected chapter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse EPUBs and show selected chapters/chunks without loading a model.",
    )
    parser.add_argument(
        "--print-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print each translated chunk while translating. Enabled by default.",
    )
    parser.add_argument(
        "--print-source-text",
        action="store_true",
        help="Also print the English source chunk before translating it.",
    )
    parser.add_argument(
        "--chunk-preview-chars",
        type=int,
        default=DEFAULT_CHUNK_PREVIEW_CHARS,
        help="Max characters to print per source/translation chunk. Use 0 for full text.",
    )
    parser.add_argument(
        "--keep-source-on-error",
        action="store_true",
        help="Keep the source chunk if translation fails. Default: fail the current model run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Hard max batch size for adapters that support batched generation.",
    )
    parser.add_argument(
        "--batch-token-limit",
        type=int,
        default=DEFAULT_BATCH_TOKEN_LIMIT,
        help=(
            "Approximate max input tokens per generated batch. Use 0 to disable "
            "dynamic token batching and rely only on --batch-size."
        ),
    )
    parser.add_argument(
        "--cache-translations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache deterministic draft/QE model chunk translations on disk.",
    )
    parser.add_argument(
        "--translation-cache-dir",
        default=str(DEFAULT_TRANSLATION_CACHE_DIR),
        help="Directory used for cached chunk translations.",
    )
    parser.add_argument(
        "--refresh-translation-cache",
        action="store_true",
        help="Ignore existing cached translations and rewrite them after translating.",
    )
    parser.add_argument(
        "--dynamic-batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group chunks by token budget when the adapter supports batching.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Previous/next chunk count passed to LLM patchers.",
    )
    parser.add_argument(
        "--memo-sample-chunks",
        type=int,
        default=12,
        help="Max source/draft samples used to build the chapter style memo.",
    )
    parser.add_argument(
        "--memo-token-budget",
        type=int,
        default=600,
        help="Max approximate tokens kept in the style memo injected into every patcher call.",
    )
    parser.add_argument(
        "--draft-n-best",
        type=int,
        default=4,
        help="Number of beam hypotheses to generate per chunk in CT2 draft models; best is kept.",
    )
    parser.add_argument(
        "--group-short-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Translate consecutive very short chunks together, then split them back by sentinels.",
    )
    parser.add_argument(
        "--short-chunk-chars",
        type=int,
        default=60,
        help="Maximum stripped source length for short-chunk grouping.",
    )
    parser.add_argument(
        "--qe-candidate-models",
        default="auto",
        help="Comma-separated extra models for --patcher qe. Use auto for NLLB/OPUS or MADLAD/OPUS.",
    )
    parser.add_argument(
        "--prefer-ct2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer CTranslate2 NLLB/OPUS candidate models in automatic QE mode.",
    )
    parser.add_argument(
        "--qe-engine",
        choices=["auto", "cometkiwi", "heuristic"],
        default="auto",
        help="Quality-estimation engine for --patcher qe.",
    )
    parser.add_argument(
        "--comet-model",
        default=DEFAULT_COMET_MODEL,
        help=(
            "COMET QE model used when --qe-engine is cometkiwi or auto. "
            "Use auto to try COMETKiwi first, then open COMET-QE."
        ),
    )
    parser.add_argument(
        "--comet-batch-size",
        type=int,
        default=8,
        help="Batch size for COMETKiwi scoring.",
    )
    parser.add_argument(
        "--comet-gpus",
        type=int,
        default=1,
        help="GPU count passed to COMET predict. Use 0 to force CPU scoring.",
    )
    parser.add_argument(
        "--quiet-comet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress COMET/PyTorch Lightning console noise during QE scoring.",
    )
    parser.add_argument(
        "--comet-fallback-heuristic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to local heuristic QE if COMETKiwi cannot load or score.",
    )
    parser.add_argument(
        "--qe-low-score-threshold",
        type=float,
        default=DEFAULT_PATCH_SCORE_THRESHOLD,
        help="Log selected QE candidates below this raw COMETKiwi score.",
    )
    parser.add_argument(
        "--selective-patching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch only chunks with low QE/local validation signals in LLM patch modes.",
    )
    parser.add_argument(
        "--patch-score-threshold",
        type=float,
        default=DEFAULT_PATCH_SCORE_THRESHOLD,
        help="COMETKiwi score below which a draft chunk is sent to the LLM patcher.",
    )
    parser.add_argument(
        "--heuristic-patch-threshold",
        type=float,
        default=DEFAULT_HEURISTIC_PATCH_THRESHOLD,
        help="Fallback heuristic score below which a draft chunk is sent to the LLM patcher.",
    )
    parser.add_argument(
        "--skip-patch-heuristic-threshold",
        type=float,
        default=96.0,
        help="Skip LLM patching for drafts above this local score unless they have hard validation flags.",
    )
    parser.add_argument(
        "--patcher-sample-candidates",
        type=int,
        default=2,
        help="Number of LLM patch candidates to generate per selected chunk before reranking.",
    )
    parser.add_argument(
        "--patcher-sample-temperature",
        type=float,
        default=0.3,
        help="Temperature for extra sampled LLM patch candidates after the deterministic first pass.",
    )
    parser.add_argument(
        "--patch-accept-margin",
        type=float,
        default=8.0,
        help=(
            "Reject an LLM/QE patch if local validation scores it this many points "
            "below the draft. Set 0 for strict no-worse local acceptance."
        ),
    )
    parser.add_argument(
        "--llm-chapter-memo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let the patcher refine the deterministic chapter memo before chunk patching.",
    )
    parser.add_argument(
        "--literary-critic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In literary mode, ask the patcher to critique meaning preservation before accepting edits.",
    )
    parser.add_argument(
        "--literary-reread",
        choices=["off", "failed", "dialogue", "all"],
        default="failed",
        help=(
            "In literary mode, run a chapter reread pass over failed chunks, dialogue "
            "chunks, every chunk, or skip it."
        ),
    )
    parser.add_argument(
        "--reread-window",
        type=int,
        default=3,
        help="Previous/next chunk count passed to the literary reread pass.",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=2,
        help="Maximum targeted repair attempts for a rejected literary edit.",
    )
    parser.add_argument(
        "--literary-arbitration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In literary mode, create a second draft for risky chunks so the editor "
            "can compare two machine translations before rewriting."
        ),
    )
    parser.add_argument(
        "--arbitration-model",
        choices=["nllb", "nllb-ct2", "opus", "opus-ct2", "madlad", "madlad-ct2", "seamless", "sonar"],
        default="nllb-ct2",
        help="Secondary draft model used by --literary-arbitration.",
    )
    parser.add_argument(
        "--arbitration-max-chunks",
        type=int,
        default=0,
        help="Maximum risky chunks to send to the arbitration model. Use 0 for no cap.",
    )
    parser.add_argument(
        "--back-translation-check",
        choices=["off", "changed", "failed", "all"],
        default="changed",
        help=(
            "In literary mode, back-translate edited chunks with the loaded editor "
            "and reject/repair likely meaning drift."
        ),
    )
    parser.add_argument(
        "--back-translation-threshold",
        type=float,
        default=0.46,
        help="Minimum source/back-translation content overlap before repair is requested.",
    )
    parser.add_argument(
        "--strict-acceptance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject post-edits that introduce hard validator failures or score below the safer draft.",
    )
    parser.add_argument(
        "--correction-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use and update a JSONL memory of accepted literary corrections.",
    )
    parser.add_argument(
        "--correction-memory-dir",
        default=str(DEFAULT_CORRECTION_MEMORY_DIR),
        help="Directory for accepted correction memory JSONL files.",
    )
    parser.add_argument(
        "--correction-memory-examples",
        type=int,
        default=3,
        help="Maximum similar accepted corrections included in editor prompts.",
    )
    parser.add_argument(
        "--correction-memory-threshold",
        type=float,
        default=0.56,
        help="Minimum similarity for correction memory examples.",
    )
    parser.add_argument(
        "--correction-memory-max-entries",
        type=int,
        default=5000,
        help="Maximum recent correction memory entries loaded per run.",
    )
    parser.add_argument(
        "--book-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist automatic book-level glossary/style/register memory between chapters.",
    )
    parser.add_argument(
        "--book-memory-dir",
        default=str(DEFAULT_BOOK_MEMORY_DIR),
        help="Directory for automatic book-level memory JSON files.",
    )
    parser.add_argument(
        "--glossary-enforcement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate preferred/forbidden glossary terminology during literary QA.",
    )
    parser.add_argument(
        "--candidate-reranking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rerank draft, alternate drafts, memory hits, and LLM edits before accepting a chunk.",
    )
    parser.add_argument(
        "--literary-window-reread",
        choices=["off", "failed", "dialogue", "all"],
        default="dialogue",
        help="Run strict JSON paragraph-window editing after chunk-level literary passes.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Number of adjacent chunks sent to each JSON window edit.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=3,
        help="Stride for all-chapter window editing.",
    )
    parser.add_argument(
        "--window-max-windows",
        type=int,
        default=0,
        help="Maximum JSON window edits per chapter. Use 0 for no cap.",
    )
    parser.add_argument(
        "--llm-json-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require schema-like JSON arrays for multi-chunk window edits.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "default", "eager", "sdpa", "flash_attention_2"],
        default="auto",
        help=(
            "Transformers attention backend. 'auto' (default) uses flash_attention_2 "
            "when flash-attn is installed, otherwise sdpa."
        ),
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply torch.compile to loaded Transformers models when supported (default: on).",
    )
    parser.add_argument(
        "--fused-models",
        action="store_true",
        default=False,
        help=(
            "Keep draft and patcher models loaded simultaneously instead of unloading the "
            "draft before loading the patcher. Requires enough VRAM for both models."
        ),
    )
    parser.add_argument(
        "--book-config",
        default="",
        help=(
            "Path to a TOML book configuration file that overrides character aliases, "
            "genders, relationships, and glossary rules. "
            "(Generate a template: python -m lib.book_config)"
        ),
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="Mode passed to torch.compile when --torch-compile is enabled.",
    )
    parser.add_argument(
        "--torch-compile-backend",
        default="",
        help="Optional backend passed to torch.compile, for example inductor.",
    )
    parser.add_argument(
        "--torch-compile-strict",
        action="store_true",
        help="Fail the run if torch.compile fails instead of continuing uncompiled.",
    )
    parser.add_argument(
        "--french-typography",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply lightweight French punctuation cleanup before writing output.",
    )
    parser.add_argument(
        "--summary-log",
        default=str(DEFAULT_SUMMARY_LOG),
        help="Append run timing summaries to this JSONL file.",
    )
    parser.add_argument(
        "--evaluation-dir",
        default=str(DEFAULT_EVALUATION_DIR),
        help="Directory for JSON quality reports from literary/benchmark runs.",
    )
    parser.add_argument(
        "--eval-reference",
        default="",
        help=(
            "Optional reference EPUB used only for evaluation reports. It is never "
            "included in generation prompts."
        ),
    )
    parser.add_argument(
        "--error-log",
        default=str(DEFAULT_ERROR_LOG),
        help="Append full error details to this file. Default: logs/translation_errors.log.",
    )
    parser.add_argument(
        "--no-error-log",
        action="store_true",
        help="Disable file logging for errors.",
    )

    # New feature flags (disabled by default; enable in config.toml or on CLI)
    parser.add_argument(
        "--auto-glossary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extract glossary terms from chapter 1 source text automatically via the patcher LLM.",
    )
    parser.add_argument(
        "--entity-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Track named entity translations across chapters for consistency (requires spaCy).",
    )
    parser.add_argument(
        "--coref-resolution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use fastcoref to improve speaker attribution in ambiguous dialogue chunks.",
    )
    parser.add_argument(
        "--chrf-evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute chrF score against --eval-reference at the end of each chapter.",
    )
    parser.add_argument(
        "--chapter-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-chunk progress to a sidecar .jsonl so interrupted chapters can resume.",
    )
    parser.add_argument(
        "--persistent-style-memo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Accumulate the LLM style memo across chapters (requires --book-memory).",
    )
    parser.add_argument(
        "--french-nlp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable spaCy fr_core_news_sm checks per chunk: tense mixing (passé simple "
            "vs passé composé), gender/number agreement. Requires: "
            "pip install spacy && python -m spacy download fr_core_news_sm"
        ),
    )
    parser.add_argument(
        "--grammar-check",
        action=argparse.BooleanOptionalAction,
        default=False,  # enable via --grammar-check
        help=(
            "Run LanguageTool grammar checking on final French output and log issues. "
            "Requires a local LanguageTool server (configure url via --lang-config TOML file)."
        ),
    )
    parser.add_argument(
        "--lang-config",
        default="french",
        help="Language config name or path (default: french). Override default patterns via a custom TOML file.",
    )

    # Load global config.toml defaults before parsing so CLI can override them.
    _cfg_path = Path("config.toml")
    if _cfg_path.exists():
        try:
            try:
                import tomllib as _toml
            except ImportError:
                import tomli as _toml  # type: ignore[no-redef]
            _cfg_defs = _toml.loads(_cfg_path.read_text(encoding="utf-8")).get("defaults", {})
            _key_map = {
                "model": "model", "strategy": "strategy", "chapter": "chapter",
                "book_config": "book_config", "batch_size": "batch_size",
                "batch_token_limit": "batch_token_limit", "context_window": "context_window",
                "attn_implementation": "attn_implementation",
                "torch_compile": "torch_compile", "fused_models": "fused_models",
                "qe_engine": "qe_engine", "comet_model": "comet_model",
                "auto_glossary": "auto_glossary", "entity_memory": "entity_memory",
                "coref_resolution": "coref_resolution", "chrf_evaluation": "chrf_evaluation",
                "chapter_checkpoint": "chapter_checkpoint",
                "persistent_style_memo": "persistent_style_memo",
                "grammar_check": "grammar_check",
                "print_text": "print_text", "chunk_preview_chars": "chunk_preview_chars",
            }
            _overrides = {
                dest: _cfg_defs[key]
                for key, dest in _key_map.items()
                if key in _cfg_defs
            }
            if _overrides:
                parser.set_defaults(**_overrides)
        except Exception:
            pass

    args = parser.parse_args()

    # Merge positional epub_files into args.book
    if args.epub_files:
        args.book = (args.book or []) + args.epub_files
    del args.epub_files

    if not args.book:
        parser.error("Provide at least one EPUB file (positional or via --book).")

    return args


