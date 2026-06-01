"""Text normalisation and display utilities."""
from __future__ import annotations
import re, unicodedata
from lib.constants import KEYBOARD_REPLACEMENTS, ONES, TENS, TRANSLATION_STOPWORDS, ENGLISH_MARKERS, FRENCH_MARKERS

def normalize_text(value: str) -> str:
    value = value.lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def number_to_words(value: int) -> str | None:
    if value in ONES:
        return ONES[value]
    if value in TENS:
        return TENS[value]
    if 20 < value < 100:
        tens = value - (value % 10)
        return f"{TENS[tens]} {ONES[value % 10]}"
    return None


def words_to_number(value: str) -> int | None:
    normalized = normalize_text(value)
    inverse_ones = {word: number for number, word in ONES.items()}
    inverse_tens = {word: number for number, word in TENS.items()}

    if normalized.isdigit():
        return int(normalized)
    if normalized in inverse_ones:
        return inverse_ones[normalized]
    if normalized in inverse_tens:
        return inverse_tens[normalized]

    parts = normalized.split()
    if len(parts) == 2 and parts[0] in inverse_tens and parts[1] in inverse_ones:
        return inverse_tens[parts[0]] + inverse_ones[parts[1]]
    return None


def requested_chapter_number(chapter: str) -> int | None:
    normalized = normalize_text(chapter)
    for prefix in ("chapter ", "chap ", "ch "):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix).strip()
            break
    return words_to_number(normalized)


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


def extract_chapter_number(value: str) -> int | None:
    normalized = normalize_text(value)
    match = re.search(r"\b(?:chapter|chap|ch)\s+(.+)$", normalized)
    if not match:
        return None

    tokens = match.group(1).split()
    for width in (2, 1):
        number = words_to_number(" ".join(tokens[:width]))
        if number is not None:
            return number
    return None

def should_translate(text: str) -> bool:
    return len(text.strip()) >= 2 and re.search(r"[A-Za-z]", text) is not None


def word_tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z\u00c0-\u00ff']+", value.lower())


def french_typography(value: str) -> str:
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = re.sub(r'"([^"\n]+)"', r"« \1 »", value)
    value = re.sub(r"\s+([,.])", r"\1", value)
    value = re.sub(r"\s+([;:!?])", r" \1", value)
    value = re.sub(r"([\u00ab])\s+", r"\1 ", value)
    value = re.sub(r"\s+([\u00bb])", r" \1", value)
    value = re.sub(r"\b([ldjmnstc\u00e7])'\s+", r"\1'", value, flags=re.IGNORECASE)
    value = re.sub(r"\s{2,}", " ", value)
    return value.strip()


def progress_text(value: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()} ... [{len(normalized) - max_chars} more chars]"


def sanitize_keyboard_text(value: str) -> str:
    value = value.translate(str.maketrans(KEYBOARD_REPLACEMENTS))
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", ascii_text)


def maybe_ascii(value: str, enabled: bool) -> str:
    return sanitize_keyboard_text(value) if enabled else value


def ui_text(value: object) -> str:
    return sanitize_keyboard_text(str(value))


def log_text(value: object) -> str:
    return ui_text(value)

def safe_output_stem(value: str) -> str:
    sanitized = sanitize_keyboard_text(value)
    sanitized = re.sub(r"[^A-Za-z0-9._ -]+", "_", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip(" ._")
    return sanitized or "book"


def append_summary(args: argparse.Namespace, summary: dict[str, object]) -> None:
    if not args.summary_log:
        return
    path = Path(args.summary_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
