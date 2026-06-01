"""Model registries and pipeline-wide default constants."""
from __future__ import annotations
from pathlib import Path

DIRECT_MODEL_MODULES = {
    "gemma": "translators.translatorGemma",
    "madlad": "translators.translatorMADLAD",
    "madlad-3b": "translators.translatorMADLAD3B",
    "madlad-ct2": "translators.translatorMADLADCT2",
    "madlad-q4": "translators.translatorMADLADQ4",
    "madlad-spec": "translators.translatorMADLADSpec",
    "mistral": "translators.translatorMistral",
    "mistral-awq": "translators.translatorMistralAWQ",
    "mistral-vllm": "translators.translatorMistralvLLM",
    "nllb": "translators.translatorNLLB",
    "nllb-ct2": "translators.translatorNLLBCT2",
    "opus": "translators.translatorOPUS",
    "opus-ct2": "translators.translatorOPUSCT2",
    "qwen": "translators.translatorQwen",
    "qwen-awq": "translators.translatorQwenAWQ",
    "qwen-vllm": "translators.translatorQwenvLLM",
    "seamless": "translators.translatorSeamless",
    "sonar": "translators.translatorSONAR",
    "tower": "translators.translatorTower",
}
PATCHER_MODULES = {
    "mistral": "translators.translatorMistral",
    "mistral-awq": "translators.translatorMistralAWQ",
    "mistral-vllm": "translators.translatorMistralvLLM",
    "qwen": "translators.translatorQwen",
    "qwen-awq": "translators.translatorQwenAWQ",
    "qwen-vllm": "translators.translatorQwenvLLM",
    "tower": "translators.translatorTower",
}
COMPOSITE_MODELS = {
    # MADLAD-CT2 (fast, recommended default)
    "madlad-ct2-literary": ("madlad-ct2", "qwen"),
    "madlad-ct2-mistral": ("madlad-ct2", "mistral"),
    "madlad-ct2-mistral-awq": ("madlad-ct2", "mistral-awq"),
    "madlad-ct2-qe": ("madlad-ct2", "qe"),
    "madlad-ct2-qwen": ("madlad-ct2", "qwen"),
    "madlad-ct2-qwen-awq": ("madlad-ct2", "qwen-awq"),
    "madlad-ct2-tower": ("madlad-ct2", "tower"),
    # MADLAD-10B HF (slower, higher quality draft)
    "madlad-literary": ("madlad", "qwen"),
    "madlad-mistral": ("madlad", "mistral"),
    "madlad-q4-literary": ("madlad-q4", "qwen"),
    "madlad-q4-mistral": ("madlad-q4", "mistral"),
    "madlad-q4-qwen": ("madlad-q4", "qwen"),
    "madlad-qe": ("madlad", "qe"),
    "madlad-qwen": ("madlad", "qwen"),
    "madlad-tower": ("madlad", "tower"),
    # NLLB-CT2 (fastest draft, lower quality)
    "nllb-ct2-literary": ("nllb-ct2", "qwen"),
    "nllb-ct2-mistral": ("nllb-ct2", "mistral"),
    "nllb-ct2-mistral-awq": ("nllb-ct2", "mistral-awq"),
    "nllb-ct2-qe": ("nllb-ct2", "qe"),
    "nllb-ct2-qwen": ("nllb-ct2", "qwen"),
    "nllb-ct2-qwen-awq": ("nllb-ct2", "qwen-awq"),
    "nllb-ct2-tower": ("nllb-ct2", "tower"),
    "nllb-mistral": ("nllb", "mistral"),
    "nllb-qe": ("nllb", "qe"),
    "nllb-qwen": ("nllb", "qwen"),
    "nllb-tower": ("nllb", "tower"),
    # ALMA-7B-R (literary-specialist, NF4 4-bit, ~4GB VRAM)
    "alma-7b-r-qwen": ("alma-7b-r", "qwen"),
    "alma-7b-r-qwen-14b-awq": ("alma-7b-r", "qwen-14b-awq"),
    "alma-7b-r-mistral": ("alma-7b-r", "mistral"),
    "alma-7b-r-qwen-awq": ("alma-7b-r", "qwen-awq"),
    # MADLAD-CT2 + improved Qwen-14B-AWQ patcher
    "madlad-ct2-qwen-14b-awq": ("madlad-ct2", "qwen-14b-awq"),
    # NLLB-CT2 + improved Qwen-14B-AWQ patcher
    "nllb-ct2-qwen-14b-awq": ("nllb-ct2", "qwen-14b-awq"),
}
LITERARY_MODELS = {"madlad-literary", "madlad-ct2-literary", "madlad-q4-literary"}

# Auto-discover adapters in translators/ and add them to the registries.
try:
    from lib.translator_registry import discover_translators as _discover
    DIRECT_MODEL_MODULES, PATCHER_MODULES = _discover(DIRECT_MODEL_MODULES, PATCHER_MODULES)
except Exception:
    pass
DEFAULT_CHUNK_PREVIEW_CHARS = 1200
DEFAULT_ERROR_LOG = Path("logs") / "translation_errors.log"
DEFAULT_SUMMARY_LOG = Path("logs") / "run_summary.jsonl"
DEFAULT_TRANSLATION_CACHE_DIR = Path("cache") / "translations"
DEFAULT_CORRECTION_MEMORY_DIR = Path("cache") / "correction_memory"
DEFAULT_BOOK_MEMORY_DIR = Path("cache") / "book_memory"
DEFAULT_EVALUATION_DIR = Path("logs") / "evaluation"
DEFAULT_BATCH_TOKEN_LIMIT = 1400
TRANSLATION_CACHE_VERSION = 2
PIPELINE_CACHE_VERSION = 1
CORRECTION_MEMORY_VERSION = 1
DEFAULT_COMET_MODEL = "auto"
DEFAULT_COMET_MODEL_CANDIDATES = (
    "Unbabel/wmt22-cometkiwi-da",
    "Unbabel/wmt20-comet-qe-da",
)
DEFAULT_PATCH_SCORE_THRESHOLD = 0.78      # raised from 0.72 — skip more chunks with decent QE
DEFAULT_HEURISTIC_PATCH_THRESHOLD = 94.0  # raised from 92.0
PROGRESS_WIDTH = 88
INLINE_TAGS = {"b", "em", "i", "strong"}

