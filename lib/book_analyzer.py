"""Automatic book analysis for translation-quality enhancement.

Parses an EPUB without any manual configuration and extracts:
- Character names, genders, relationship types, and affinity matrix
- Narrative POV (first / third person)
- Dominant tense (past / present)
- Dialogue density
- Recurring locations and organisations
- Domain vocabulary (sports, romance, fantasy, …)
- Recurring phrase candidates for the glossary
- Per-character speech register hints

Results are cached to ``analysis/{book_slug}.json`` and re-used on subsequent
runs.  Pass ``force=True`` to bypass the cache.

Requires: ebooklib, beautifulsoup4  (already listed in requirements.txt)
Optional: spacy + en_core_web_sm  (enables NER for richer entity extraction)
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ANALYSIS_DIR = Path("analysis")

# ---------------------------------------------------------------------------
# Domain vocabulary fingerprints
# ---------------------------------------------------------------------------

_DOMAIN_VOCAB: dict[str, frozenset[str]] = {
    "sports": frozenset({
        "game","team","player","score","goal","field","court","arena","coach",
        "season","penalty","referee","league","tournament","championship","match",
        "hockey","basketball","football","soccer","baseball","tennis","swimming",
        "puck","stick","skate","shot","pass","defence","offense","goalie","rink",
    }),
    "romance": frozenset({
        "love","kiss","heart","relationship","boyfriend","girlfriend","date",
        "attraction","romance","feelings","embrace","touch","desire","longing",
        "wedding","marriage","breakup","jealous","jealousy","flirt","intimate",
    }),
    "thriller": frozenset({
        "murder","kill","weapon","detective","suspect","crime","police","escape",
        "chase","threat","danger","secret","conspiracy","betrayal","blackmail",
    }),
    "fantasy": frozenset({
        "magic","wizard","dragon","kingdom","spell","quest","enchant","potion",
        "sword","castle","elf","dwarf","orc","prophecy","ancient","realm","guild",
    }),
    "scifi": frozenset({
        "spaceship","robot","android","alien","planet","galaxy","laser","quantum",
        "cybernetic","nanite","hyperspace","singularity","terraforming",
    }),
}

# Pronouns used for gender inference
_MALE_RE = re.compile(r"\b(he|him|his|himself)\b", re.IGNORECASE)
_FEMALE_RE = re.compile(r"\b(she|her|hers|herself)\b", re.IGNORECASE)

# Non-sentence-start capitalised words
_CAPITALISED_RE = re.compile(r"(?<=\s)([A-Z][a-z]{2,})")

# Intimate relationship context
_INTIMATE_CONTEXT_RE = re.compile(
    r"\b(kiss(?:ed|ing)?|hug(?:ged|ging)?|held|touch(?:ed|ing)?|"
    r"lov(?:ed|ing)|together|boyfriend|girlfriend|partner|wife|husband|"
    r"intimat|romanc|affair|heart)\b",
    re.IGNORECASE,
)

# POV markers
_FIRST_PERSON_RE = re.compile(r"\b(I said|I thought|I felt|I was|I walked|I looked)\b", re.IGNORECASE)
_THIRD_PERSON_RE = re.compile(r"\b(he said|she said|they said|he thought|she thought)\b", re.IGNORECASE)

# Tense markers
_PAST_RE = re.compile(
    r"\b(said|was|were|had|felt|walked|looked|smiled|laughed|ran|knew|thought|came|went)\b",
    re.IGNORECASE,
)
_PRESENT_RE = re.compile(
    r"\b(says|is|are|has|feels|walks|looks|smiles|laughs|runs|knows|thinks|comes|goes)\b",
    re.IGNORECASE,
)

# Location context: "in/at/to [Place]"
_LOCATION_CONTEXT_RE = re.compile(
    r"\b(?:in|at|to|from|near|outside|inside|through|across)\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9_-]", "_", text.lower()).strip("_")[:80]


def _extract_blocks(epub_path: Path) -> tuple[list[str], dict]:
    """Return (paragraph_blocks, epub_metadata)."""
    from ebooklib import epub, ITEM_DOCUMENT  # type: ignore[import]
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(epub_path))
    meta = {
        "title": book.title or "",
        "author": "; ".join(a[0] for a in (book.creators or []) if a),
        "language": book.language or "en",
    }
    blocks: list[str] = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        for tag in soup.find_all(["p", "div"]):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 25:
                blocks.append(text)
    return blocks, meta


def _candidate_names(text: str) -> set[str]:
    """Capitalized words that appear mid-sentence — likely proper nouns."""
    try:
        from lib.constants import NAME_STOPWORDS
    except Exception:
        NAME_STOPWORDS = frozenset()  # type: ignore[assignment]

    candidates: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = sentence.split()
        for i, word in enumerate(words):
            if i == 0:
                continue
            clean = re.sub(r"[^A-Za-z\-']", "", word)
            if (
                len(clean) >= 3
                and clean[0].isupper()
                and not clean.isupper()
                and clean not in NAME_STOPWORDS
            ):
                candidates.add(clean)
    return candidates


def _count_freq(text: str, names: set[str]) -> Counter:
    return Counter({n: len(re.findall(rf"\b{re.escape(n)}\b", text)) for n in names})


def _infer_genders(text: str, chars: set[str]) -> dict[str, str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    counts: dict[str, dict[str, int]] = {c: {"m": 0, "f": 0} for c in chars}
    for i, sent in enumerate(sentences):
        window = " ".join(sentences[max(0, i - 1): i + 2])
        for name in chars:
            if not re.search(rf"\b{re.escape(name)}\b", window):
                continue
            counts[name]["m"] += len(_MALE_RE.findall(window))
            counts[name]["f"] += len(_FEMALE_RE.findall(window))
    genders: dict[str, str] = {}
    for name, c in counts.items():
        if c["m"] > c["f"] * 1.5:
            genders[name] = "male"
        elif c["f"] > c["m"] * 1.5:
            genders[name] = "female"
        else:
            genders[name] = "neutral"
    return genders


def _build_affinity(blocks: list[str], chars: set[str]) -> dict[str, dict[str, float]]:
    """Normalised co-occurrence over a +/-1 paragraph window."""
    names = sorted(chars)
    cooc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    solo: Counter = Counter()
    for i, block in enumerate(blocks):
        window = " ".join(blocks[max(0, i - 1): i + 2])
        present = [n for n in names if re.search(rf"\b{re.escape(n)}\b", window)]
        for n in present:
            solo[n] += 1
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                cooc[present[a]][present[b]] += 1
                cooc[present[b]][present[a]] += 1
    result: dict[str, dict[str, float]] = {}
    for a in names:
        result[a] = {}
        for b in names:
            if a == b:
                result[a][b] = 1.0
                continue
            denom = math.sqrt(solo[a] * solo[b]) if solo[a] and solo[b] else 0
            result[a][b] = round(cooc[a][b] / denom, 3) if denom else 0.0
    return result


def _infer_relationships(
    affinity: dict[str, dict[str, float]], text: str, chars: set[str]
) -> dict[str, str]:
    names = sorted(chars)
    relationships: dict[str, str] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            score = affinity.get(a, {}).get(b, 0.0)
            if score <= 0.0:
                continue
            intimate = bool(
                re.search(
                    rf"\b{re.escape(a)}\b.{{0,80}}{_INTIMATE_CONTEXT_RE.pattern}",
                    text,
                    re.IGNORECASE,
                )
                or re.search(
                    rf"\b{re.escape(b)}\b.{{0,80}}{_INTIMATE_CONTEXT_RE.pattern}",
                    text,
                    re.IGNORECASE,
                )
            )
            if intimate or score >= 0.55:
                rel = "intimate_partner"
            elif score >= 0.25:
                rel = "friends"
            elif score >= 0.08:
                rel = "acquaintance"
            else:
                continue
            relationships[f"{a}|{b}"] = rel
            relationships[f"{b}|{a}"] = rel
    return relationships


def _infer_aliases(
    text: str, all_candidates: set[str], canonical: set[str]
) -> dict[str, str]:
    """Map non-canonical candidates to their most frequent co-occurring character."""
    aliases: dict[str, str] = {c: c for c in canonical}
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for cand in all_candidates - canonical:
        co: Counter = Counter()
        for sent in sentences:
            if not re.search(rf"\b{re.escape(cand)}\b", sent):
                continue
            for ch in canonical:
                if re.search(rf"\b{re.escape(ch)}\b", sent):
                    co[ch] += 1
        if not co:
            continue
        top, count = co.most_common(1)[0]
        if count / sum(co.values()) > 0.6 and count >= 3:
            aliases[cand] = top
    return aliases


def _infer_dialogue_register(
    text: str, relationships: dict[str, str]
) -> dict[str, str]:
    """Per-pair dialogue register (informal / formal / unknown)."""
    register: dict[str, str] = {}
    for key, rel in relationships.items():
        if "|" not in key:
            continue
        if rel == "intimate_partner":
            register[key] = "informal"
        elif rel in {"friends", "acquaintance"}:
            register[key] = "informal"
        else:
            register[key] = "formal"
    return register


def _detect_pov(text: str) -> str:
    sample = text[:30_000]
    fp = len(_FIRST_PERSON_RE.findall(sample))
    tp = len(_THIRD_PERSON_RE.findall(sample))
    if fp > tp * 1.5:
        return "first_person"
    return "third_person"


def _detect_tense(text: str) -> str:
    sample = text[:50_000]
    past = len(_PAST_RE.findall(sample))
    present = len(_PRESENT_RE.findall(sample))
    return "past" if past >= present else "present"


def _dialogue_ratio(blocks: list[str]) -> float:
    dialogue_blocks = sum(
        1 for b in blocks if '"' in b or "“" in b or "«" in b
    )
    return round(dialogue_blocks / max(len(blocks), 1), 2)


def _extract_locations(text: str, chars: set[str]) -> list[str]:
    """Proper nouns in location context, excluding known character names."""
    found: Counter = Counter()
    for match in _LOCATION_CONTEXT_RE.finditer(text):
        place = match.group(1).strip()
        if place not in chars and len(place) >= 3:
            found[place] += 1
    return [p for p, c in found.most_common(20) if c >= 2]


def _extract_organisations(text: str, blocks: list[str], chars: set[str]) -> list[str]:
    """Detect organisation-like proper nouns (all-caps or 'the X' patterns)."""
    found: Counter = Counter()
    # All-caps words 2-6 chars (acronyms like NHL, FBI)
    for m in re.finditer(r"\b([A-Z]{2,6})\b", text):
        found[m.group(1)] += 1
    # "the [Capitalized Phrase]" patterns
    for m in re.finditer(r"\bthe\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text):
        phrase = m.group(1)
        if phrase not in chars:
            found[phrase] += 1
    return [o for o, c in found.most_common(20) if c >= 3 and o not in chars]


def _detect_domain(text: str) -> dict[str, list[str]]:
    """Return detected domain vocabulary per category."""
    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    result: dict[str, list[str]] = {}
    for domain, vocab in _DOMAIN_VOCAB.items():
        hits = sorted(words & vocab)
        if len(hits) >= 3:
            result[domain] = hits[:20]
    return result


def _recurring_phrases(blocks: list[str], min_count: int = 4) -> list[dict[str, object]]:
    """Find recurring bigrams and trigrams (candidates for glossary entries)."""
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()
    for block in blocks:
        words = re.findall(r"\b[a-z]{2,}\b", block.lower())
        for i in range(len(words) - 1):
            bigrams[f"{words[i]} {words[i+1]}"] += 1
        for i in range(len(words) - 2):
            trigrams[f"{words[i]} {words[i+1]} {words[i+2]}"] += 1
    try:
        from lib.constants import TRANSLATION_STOPWORDS
    except Exception:
        TRANSLATION_STOPWORDS = frozenset()  # type: ignore[assignment]
    phrases = []
    for phrase, count in {**bigrams, **trigrams}.items():
        if count < min_count:
            continue
        first_word = phrase.split()[0]
        if first_word in TRANSLATION_STOPWORDS:
            continue
        phrases.append({"phrase": phrase, "count": count})
    return sorted(phrases, key=lambda x: -x["count"])[:30]


def _build_character_profiles(
    text: str,
    blocks: list[str],
    chars: set[str],
    genders: dict[str, str],
    freq: Counter,
) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for name in chars:
        # Collect dialogue lines spoken by this character
        dialogue_words: list[int] = []
        for block in blocks:
            if re.search(
                rf"\b{re.escape(name)}\b[^.\n]{{0,60}}\b(said|asked|replied|shouted|whispered)\b",
                block,
                re.IGNORECASE,
            ):
                # Extract quoted text in this block
                for quote in re.findall(r'"([^"]{5,200})"', block):
                    dialogue_words.append(len(quote.split()))
        avg_dialogue = round(sum(dialogue_words) / max(len(dialogue_words), 1), 1)
        # Common words in dialogue
        dialogue_text = " ".join(
            q for block in blocks
            for q in re.findall(r'"([^"]{5,200})"', block)
            if re.search(rf"\b{re.escape(name)}\b", block, re.IGNORECASE)
        )
        word_freq = Counter(re.findall(r"\b[a-z]{3,}\b", dialogue_text.lower()))
        try:
            from lib.constants import TRANSLATION_STOPWORDS
        except Exception:
            TRANSLATION_STOPWORDS = frozenset()  # type: ignore[assignment]
        common_words = [
            w for w, _ in word_freq.most_common(10) if w not in TRANSLATION_STOPWORDS
        ][:5]
        profiles[name] = {
            "frequency": freq[name],
            "gender": genders.get(name, "neutral"),
            "avg_dialogue_words": avg_dialogue,
            "common_speech_words": common_words,
        }
    return profiles


def _style_notes(
    chars: set[str],
    genders: dict[str, str],
    relationships: dict[str, str],
    pov: str,
    tense: str,
    force_tu: bool,
    domains: dict[str, list[str]],
) -> str:
    parts = []
    pov_str = "First-person narrative" if pov == "first_person" else "Third-person narrative"
    parts.append(f"{pov_str}, {tense}-tense.")
    seen: set[frozenset] = set()
    for key, rel in relationships.items():
        if rel != "intimate_partner" or "|" not in key:
            continue
        a, b = key.split("|", 1)
        fs = frozenset([a, b])
        if fs in seen:
            continue
        seen.add(fs)
        parts.append(f"{a} and {b} are intimate partners (use tu/toi/te in their dialogue).")
    if force_tu:
        parts.append("Narrative uses informal second-person (tu) throughout.")
    for domain in domains:
        parts.append(f"Domain detected: {domain}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Glossary generation (replaces manual book configs)
# ---------------------------------------------------------------------------

def _generate_glossary(
    text: str,
    recurring: list[dict],
    domains: dict[str, list[str]],
    chars: set[str],
) -> dict[str, dict]:
    """Build regex-based glossary rules from corpus analysis.

    Produces entries compatible with GLOSSARY_RULES format:
      { term: { source: [...], preferred: [...], forbidden: [...] } }

    Sources:
    1. KNOWN_TRANSLATION_TERMS that appear in the book.
    2. High-frequency recurring phrases that aren't stopwords.
    3. Domain-specific terms with known French equivalents.
    4. Character names that need preservation (non-French names).
    """
    try:
        from lib.constants import KNOWN_TRANSLATION_TERMS, TRANSLATION_STOPWORDS
    except Exception:
        KNOWN_TRANSLATION_TERMS = {}  # type: ignore[assignment]
        TRANSLATION_STOPWORDS = frozenset()  # type: ignore[assignment]

    glossary: dict[str, dict] = {}
    text_lower = text.lower()

    # 1. Known translation terms present in the book
    for term, french in KNOWN_TRANSLATION_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            slug = re.sub(r"[^a-z0-9]+", "_", term).strip("_")
            glossary[slug] = {
                "source": [rf"\b{re.escape(term)}\b"],
                "preferred": [rf"\b{re.escape(french)}\b"],
                "forbidden": [],
            }

    # 2. Domain-specific terms with known French mappings
    _DOMAIN_FR: dict[str, str] = {
        "hockey": "hockey", "playoffs": "playoffs",
        "goal": "but", "penalty": "pénalité",
        "coach": "entraîneur", "season": "saison",
        "game": "match", "team": "équipe",
        "love": "amour", "kiss": "baiser",
        "heart": "coeur",
    }
    for term, french in _DOMAIN_FR.items():
        if re.search(rf"\b{term}\b", text_lower) and term not in glossary:
            glossary[term] = {
                "source": [rf"\b{re.escape(term)}s?\b"],
                "preferred": [rf"\b{re.escape(french)}s?\b"],
                "forbidden": [],
            }

    # 3. Recurring multi-word phrases (trigrams) — mark for source matching only
    for item in recurring[:10]:
        phrase = str(item["phrase"])
        words = phrase.split()
        if len(words) < 2:
            continue
        if any(w in TRANSLATION_STOPWORDS for w in words):
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", phrase).strip("_")
        if slug not in glossary:
            glossary[slug] = {
                "source": [rf"\b{re.escape(phrase)}\b"],
                "preferred": [],
                "forbidden": [],
            }

    return glossary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_book(
    epub_path: str | Path,
    cache_dir: str | Path = ANALYSIS_DIR,
    force: bool = False,
) -> dict:
    """Analyse an EPUB and return a profile dict compatible with apply_book_config_dict.

    The result is cached to ``cache_dir/{slug}.json``.  Set ``force=True`` to
    re-analyse even when a cached result exists.
    """
    epub_path = Path(epub_path)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    slug = _slug(epub_path.stem)
    cache_path = Path(cache_dir) / f"{slug}.json"

    if cache_path.exists() and not force:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Book profile loaded from cache: {cache_path}")
        return data

    print(f"Analysing '{epub_path.name}' for characters, relationships, and style…")

    try:
        blocks, meta = _extract_blocks(epub_path)
    except Exception as exc:
        print(f"WARNING: could not extract text for analysis ({exc}). Running without profile.")
        return {}

    if not blocks:
        return {}

    full_text = "\n".join(blocks)

    # --- Character detection ---
    all_candidates = _candidate_names(full_text)
    freq = _count_freq(full_text, all_candidates)
    min_freq = max(5, len(blocks) // 30)
    chars = {name for name, count in freq.items() if count >= min_freq}

    if not chars:
        print("  WARNING: no significant character names found.")
        return {}

    print(f"  Characters ({len(chars)}): {', '.join(sorted(chars)[:8])}"
          + (" …" if len(chars) > 8 else ""))

    genders = _infer_genders(full_text, chars)
    affinity = _build_affinity(blocks, chars)
    relationships = _infer_relationships(affinity, full_text, chars)
    aliases = _infer_aliases(full_text, all_candidates, chars)
    dialogue_register = _infer_dialogue_register(full_text, relationships)

    # force_tu: dominant intimate-pair relationship + heavy 2nd-person use
    second_person_hits = len(re.findall(
        r"\b(you|your|yours|yourself)\b", full_text[:50_000], re.IGNORECASE
    ))
    has_intimate = any(v == "intimate_partner" for v in relationships.values())
    force_tu = has_intimate and second_person_hits > 100

    # --- Narrative style ---
    pov = _detect_pov(full_text)
    tense = _detect_tense(full_text)
    dlg_ratio = _dialogue_ratio(blocks)

    # --- World/setting ---
    locations = _extract_locations(full_text, chars)
    organisations = _extract_organisations(full_text, blocks, chars)
    domains = _detect_domain(full_text)
    recurring = _recurring_phrases(blocks)
    char_profiles = _build_character_profiles(full_text, blocks, chars, genders, freq)

    intimate_count = sum(1 for v in relationships.values() if v == "intimate_partner") // 2
    if intimate_count:
        print(f"  Intimate pairs: {intimate_count}  |  force_tu: {force_tu}")
    print(f"  POV: {pov}  |  Tense: {tense}  |  Dialogue density: {dlg_ratio:.0%}")
    if domains:
        print(f"  Detected domains: {', '.join(domains)}")
    if locations:
        print(f"  Locations ({len(locations)}): {', '.join(locations[:5])}")

    notes = _style_notes(chars, genders, relationships, pov, tense, force_tu, domains)
    glossary = _generate_glossary(full_text, recurring, domains, chars)

    if glossary:
        print(f"  Glossary rules generated: {len(glossary)}")

    # --- Semantic analysis (emotion arc, speech acts, pragmatics) ---
    semantic_data: dict = {}
    try:
        from lib.semantics import analyze_semantics, profile_to_dict
        sem = analyze_semantics(blocks, known_chars=chars)
        semantic_data = profile_to_dict(sem)
        if sem.dominant_emotions:
            print(f"  Emotional register: {', '.join(sem.dominant_emotions[:2])}")
    except Exception as exc:
        print(f"  Semantic analysis skipped: {exc}")

    profile = {
        "book_slug": slug,
        "metadata": meta,
        "narrative": {
            "pov": pov,
            "tense": tense,
            "dialogue_ratio": dlg_ratio,
        },
        "characters": {
            "aliases": aliases,
            "genders": genders,
            "relationships": relationships,
            "profiles": char_profiles,
        },
        "affinity_matrix": affinity,
        "dialogue_register": dialogue_register,
        "register": {"force_tu": force_tu},
        "locations": locations,
        "organisations": organisations,
        "domain_vocabulary": domains,
        "recurring_phrases": recurring,
        "glossary": glossary,
        "semantics": semantic_data,
        "style_notes": notes,
    }

    cache_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Profile saved: {cache_path}")
    return profile
