"""French NLP validation using spaCy (fr_core_news_sm).

Detects the most common translation errors found in practice:
  1. Tense mixing: passé simple and passé composé in the same paragraph
  2. English word leaks: common English function words left untranslated
  3. Hallucinated numbers/dates: numbers in translation absent from source
  4. Gender/number agreement: adjective morph mismatches with head noun

None of these functions raise; they return empty lists on failure.
"""
from __future__ import annotations

import re


# spaCy model (lazy, cached)

_nlp = None


def _load_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("fr_core_news_sm")
        except OSError:
            raise RuntimeError(
                "French spaCy model not found. "
                "Run: python -m spacy download fr_core_news_sm"
            )
        return _nlp
    except ImportError as exc:
        raise RuntimeError("spacy is required. Run: pip install spacy") from exc


# 1. Tense mixing

def detect_tense_mixing(text: str) -> list[str]:
    """Return warning strings for paragraphs that mix passé simple and passé composé.

    Passé simple:  finite VERB/AUX with Tense=Past, VerbForm=Fin
    Passé composé: past participle (VerbForm=Part, Tense=Past) with an AUX parent
    """
    try:
        nlp = _load_nlp()
    except Exception:
        return []

    warnings = []
    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if len(p.strip()) > 60]
    if not paragraphs:
        paragraphs = [text]

    for para in paragraphs:
        try:
            doc = nlp(para[:2000])
        except Exception:
            continue

        ps_verbs = []   # passé simple finite forms
        pc_verbs = []   # past participles (passé composé)

        for token in doc:
            morph = token.morph
            tense = morph.get("Tense")
            verbform = morph.get("VerbForm")
            mood = morph.get("Mood")

            if token.pos_ in {"VERB", "AUX"} and "Past" in tense:
                if "Fin" in verbform:
                    # finite past = passé simple
                    ps_verbs.append(token.text)
                elif "Part" in verbform:
                    # past participle — passé composé if governed by an auxiliary
                    if token.head.pos_ == "AUX" or any(
                        c.pos_ == "AUX" for c in token.children
                    ):
                        pc_verbs.append(token.text)

        if ps_verbs and pc_verbs:
            warnings.append(
                f"tense_mixing: passé simple ({', '.join(ps_verbs[:3])}) "
                f"and passé composé ({', '.join(pc_verbs[:3])}) in same paragraph"
            )

    return warnings


# 2. English word leaks

# English function words that should never appear in French literary text
_EN_FUNCTION_WORDS = frozenset("""
the a an is are was were be been being have has had do does did
will would could should may might must shall ought
you your yours yourself him her his hers them their theirs
who what which when where why how much many some any all
but and or nor yet so because though although however therefore
oh well just like so okay yeah no yes shut up let go
""".split())

# French-looking words that happen to match English tokens — safe to ignore
_FR_FALSE_POSITIVES = frozenset("a an or so no".split())


def detect_english_leaks(source: str, translation: str) -> list[str]:
    """Return English words that appear in the translation but not in the source."""
    # Extract all-ASCII tokens from translation (potential English words)
    trans_tokens = re.findall(r"\b[a-zA-Z]{2,}\b", translation)
    source_lower = source.lower()

    leaks = []
    seen: set[str] = set()
    for tok in trans_tokens:
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        if (
            low in _EN_FUNCTION_WORDS
            and low not in _FR_FALSE_POSITIVES
            and low not in source_lower  # allow words that were in the source (names etc.)
        ):
            leaks.append(tok)

    return leaks


# 3. Hallucinated numbers / dates

_NUMBER_RE = re.compile(r"\b(\d+)\b")
_YEAR_RE   = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def detect_hallucinated_numbers(source: str, translation: str) -> list[str]:
    """Return numbers that appear in the translation but not in the source.

    Years (4-digit) are checked separately and always flagged if absent.
    """
    src_nums  = set(_NUMBER_RE.findall(source))
    trans_nums = set(_NUMBER_RE.findall(translation))
    extra = trans_nums - src_nums

    warnings = []
    for num in sorted(extra):
        if _YEAR_RE.match(num):
            warnings.append(f"hallucinated_year:{num}")
        elif int(num) > 9:   # ignore small numbers that appear everywhere
            warnings.append(f"hallucinated_number:{num}")
    return warnings


# 4. Gender / number agreement check

def detect_agreement_errors(text: str, max_errors: int = 5) -> list[str]:
    """Detect adjective–noun gender/number mismatches via spaCy dependency parse."""
    try:
        nlp = _load_nlp()
    except Exception:
        return []

    errors = []
    try:
        doc = nlp(text[:3000])
    except Exception:
        return []

    for token in doc:
        if token.pos_ != "ADJ":
            continue
        # Find the governing noun via amod or attr dependency
        head = token.head
        if head.pos_ not in {"NOUN", "PROPN"}:
            continue

        noun_morph = head.morph
        adj_morph  = token.morph

        n_gender = noun_morph.get("Gender")
        a_gender = adj_morph.get("Gender")
        n_number = noun_morph.get("Number")
        a_number = adj_morph.get("Number")

        # Only flag when both sides have explicit features and they disagree
        if n_gender and a_gender and set(n_gender) != set(a_gender):
            errors.append(
                f"gender_mismatch: '{head.text}' ({'/'.join(n_gender)}) "
                f"+ '{token.text}' ({'/'.join(a_gender)})"
            )
        if n_number and a_number and set(n_number) != set(a_number):
            errors.append(
                f"number_mismatch: '{head.text}' ({'/'.join(n_number)}) "
                f"+ '{token.text}' ({'/'.join(a_number)})"
            )
        if len(errors) >= max_errors:
            break

    return errors


# 5. Untranslated English in dialogue

# Dialogue delimiters used in French text
_DIALOGUE_RE = re.compile(
    r'(?:«\s*)(.*?)(?:\s*»)|(?:^|\n)\s*—\s*([^\n—]+)',
    re.DOTALL,
)

# English words that reveal a whole dialogue line was left untranslated
_DIALOGUE_EN_MARKERS = frozenset("""
the you your is are was were have has had will would could should
just like so not don't can't won't didn't doesn't I'm I'll I've
shut up let go come on right okay no way that's it's what's
""".split())


def detect_dialogue_english(translation: str, source: str) -> list[str]:
    """Return dialogue lines in the translation that appear to be untranslated English.

    A dialogue line is flagged when it contains at least 2 English function words
    that were NOT already present in the corresponding source dialogue.
    """
    src_dialogue_lower = " ".join(
        m.group(1) or m.group(2) or ""
        for m in _DIALOGUE_RE.finditer(source)
    ).lower()

    warnings = []
    for m in _DIALOGUE_RE.finditer(translation):
        line = (m.group(1) or m.group(2) or "").strip()
        if not line or len(line) < 3:
            continue
        # Count English function words in this line
        tokens = re.findall(r"\b[a-zA-Z']+\b", line.lower())
        en_hits = [t for t in tokens if t in _DIALOGUE_EN_MARKERS and t not in src_dialogue_lower]
        if len(en_hits) >= 2:
            warnings.append(f"untranslated_dialogue: «{line[:60]}»")
    return warnings[:3]


# 6. Sentence count ratio (omission / inflation detector)

_SENT_END_RE = re.compile(r"[.!?»]\s")


def sentence_count(text: str) -> int:
    return max(1, len(_SENT_END_RE.findall(text)))


def check_length_ratio(source: str, translation: str) -> str | None:
    """Return a warning string if the sentence count ratio is suspiciously off."""
    src_n = sentence_count(source)
    tr_n  = sentence_count(translation)
    ratio = tr_n / src_n
    if ratio < 0.6:
        return f"possible_omission: {src_n} source sentences  {tr_n} in translation (ratio {ratio:.2f})"
    if ratio > 2.0:
        return f"possible_inflation: {src_n} source sentences  {tr_n} in translation (ratio {ratio:.2f})"
    return None
