"""Static translation constants.

Defaults are embedded directly in Python below.  If ``data/translation_data.toml``
exists (user-generated override, never committed), those values take precedence.

To create a customisable TOML template run:
    python -c "from lib.constants import export_defaults; export_defaults()"
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Python defaults (source of truth)
# ---------------------------------------------------------------------------

_DEFAULT_STOPWORDS_EN = frozenset({
    "a","about","after","all","an","and","are","as","at","be","been",
    "but","by","for","from","had","has","have","he","her","him","his",
    "i","in","is","it","its","me","not","of","on","or","she","that",
    "the","their","them","then","there","they","this","to","up",
    "was","were","with","you","your",
})

_DEFAULT_NAME_STOPWORDS = frozenset({
    "Chapter","Part","Prologue","Epilogue","Interlude",
    "One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten",
    "Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen",
    "Seventeen","Eighteen","Nineteen","Twenty","Thirty","Forty","Fifty",
    "January","February","March","April","May","June","July","August",
    "September","October","November","December",
    "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday",
    "English","French","Spanish","German","Italian","Japanese","Chinese",
    "Source","Target","Memo","Note","Editor","Author","Title",
    "God","Lord","Christ","Jesus","Buddha",
    "Oh","Ok","Okay","Yeah","Yep","Nope","No","Yes","Hey","Hi","Hello",
})

_DEFAULT_ENGLISH_MARKERS = frozenset({
    "and","chapter","for","had","he","her","him","his","it",
    "july","said","she","that","the","this","was","were","with","you",
})

_DEFAULT_FRENCH_MARKERS = frozenset({
    "avec","ce","chapitre","dans","de","des","du","elle","elles",
    "en","est","et","il","ils","je","juillet","la","le","les",
    "lui","mais","ne","nous","pas","pour","que","qui","se",
    "tu","un","une","vous",
})

_DEFAULT_KNOWN_TERMS: dict[str, str] = {
    "boyfriend":   "petit ami",
    "carbs":       "glucides",
    "chapter":     "chapitre",
    "hockey":      "hockey",
    "parking lot": "parking",
    "playoffs":    "playoffs",
    "shorts":      "short",
    "trail":       "sentier",
}

_DEFAULT_MARKER_HINTS: dict[str, list[str]] = {
    "healthy":     [r"\bsains?\b", r"\bbon(?:s|nes)?\b"],
    "smoke":       [r"\bfumes?\b", r"\bfumez\b", r"\bfumer\b"],
    "last night":  [r"\bhier soir\b", r"\bla nuit derni[eè]re\b"],
    "now":         [r"\bmaintenant\b"],
    "real":        [r"\bvrai(?:e|s|es)?\b", r"\br[eé]el(?:le|s|les)?\b"],
    "that":        [r"\b[cç]a\b", r"\bcela\b", r"\bce\b", r"\bcette\b"],
}

_DEFAULT_BLOCK_TAGS = frozenset({
    "blockquote","caption","dd","dt","figcaption",
    "h1","h2","h3","h4","h5","h6","li","p","td","th",
})

_DEFAULT_SKIP_INSIDE = frozenset({"script","style","svg","math"})
_DEFAULT_INLINE_TAGS = frozenset({"b","em","i","strong"})
_DEFAULT_NON_STORY = frozenset({
    "about","acknowledg","booklist","contents","copyright","cover",
    "dedication","excerpt","intro","introduction","teaser","title","toc",
})

_DEFAULT_ONES = {
    0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",
    6:"six",7:"seven",8:"eight",9:"nine",10:"ten",
    11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",15:"fifteen",
    16:"sixteen",17:"seventeen",18:"eighteen",19:"nineteen",
}

_DEFAULT_TENS = {
    20:"twenty",30:"thirty",40:"forty",50:"fifty",
    60:"sixty",70:"seventy",80:"eighty",90:"ninety",
}


# ---------------------------------------------------------------------------
# TOML override loader (optional — data/ is gitignored)
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        import tomli as tomllib  # type: ignore[no-redef]
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except ImportError:
        return {}


_DATA_PATH = Path(__file__).parent.parent / "data" / "translation_data.toml"
_d = _load_toml(_DATA_PATH)


def _get(section: str, key: str, default):
    return _d.get(section, {}).get(key, default if not _d else default)


# ---------------------------------------------------------------------------
# Public constants (TOML override merged with Python defaults)
# ---------------------------------------------------------------------------

TRANSLATION_STOPWORDS: frozenset[str] = frozenset(
    _d.get("stopwords", {}).get("english", None) or _DEFAULT_STOPWORDS_EN
)
NAME_STOPWORDS: frozenset[str] = frozenset(
    _d.get("stopwords", {}).get("names", None) or _DEFAULT_NAME_STOPWORDS
)
ENGLISH_MARKERS: frozenset[str] = frozenset(
    _d.get("markers", {}).get("english", None) or _DEFAULT_ENGLISH_MARKERS
)
FRENCH_MARKERS: frozenset[str] = frozenset(
    _d.get("markers", {}).get("french", None) or _DEFAULT_FRENCH_MARKERS
)
KNOWN_TRANSLATION_TERMS: dict[str, str] = dict(
    _d.get("known_terms", None) or _DEFAULT_KNOWN_TERMS
)
_mh = _d.get("marker_hints", None) or _DEFAULT_MARKER_HINTS
INLINE_MARKER_HINTS: dict[str, list[str]] = {k: list(v) for k, v in _mh.items()}

_html = _d.get("html", {})
BLOCK_TAGS: frozenset[str] = frozenset(_html.get("block_tags", None) or _DEFAULT_BLOCK_TAGS)
SKIP_INSIDE_TAGS: frozenset[str] = frozenset(_html.get("skip_inside_tags", None) or _DEFAULT_SKIP_INSIDE)
INLINE_TAGS: frozenset[str] = frozenset(_html.get("inline_tags", None) or _DEFAULT_INLINE_TAGS)
NON_STORY_FILENAMES: frozenset[str] = frozenset(
    _html.get("non_story_filenames", None) or _DEFAULT_NON_STORY
)

_nw = _d.get("number_words", {})
_ones_list = _nw.get("ones", None)
_tens_list = _nw.get("tens", None)
ONES: dict[int, str] = (
    {i: w for i, w in enumerate(_ones_list)} if _ones_list else dict(_DEFAULT_ONES)
)
TENS: dict[int, str] = (
    {(i + 2) * 10: w for i, w in enumerate(_tens_list)} if _tens_list else dict(_DEFAULT_TENS)
)

PROMPT_LEAK_MARKERS: tuple[str, ...] = (
    "English:",
    "French:",
    "Translate the following",
    "Return only",
    "Here is",
    "<|im_start|>",
    "<|im_end|>",
)

KEYBOARD_REPLACEMENTS: dict[str, str] = {
    " ": " ",   # non-breaking space
    "©": "(c)",
    "«": '"',
    "­": "-",   # soft hyphen
    "®": "(r)",
    "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "'": "'", "'": "'", "‚": ",", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "•": "-", "…": "...", "′": "'", "″": '"',
    "€": "EUR", "™": "(tm)",
    "←": "<-", "→": "->", "⇒": "=>",
    "✓": "[OK]", "✔": "[OK]", "❌": "[FAIL]", "⚠": "[WARN]",
    "⟦": "[[", "⟧": "]]", "×": "x",
}


# ---------------------------------------------------------------------------
# Export helper — generates data/translation_data.toml from defaults
# ---------------------------------------------------------------------------

def export_defaults(dest: Path | None = None) -> None:
    """Write a TOML template from the embedded Python defaults.

    Run:  python -c "from lib.constants import export_defaults; export_defaults()"
    """
    import json as _json
    dest = dest or (Path(__file__).parent.parent / "data" / "translation_data.toml")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _arr(items) -> str:
        return "[\n" + "".join(f'  "{i}",\n' for i in sorted(items)) + "]"

    def _table(d: dict) -> str:
        return "\n".join(f'"{k}" = "{v}"' for k, v in sorted(d.items()))

    lines = [
        "# Auto-generated from lib/constants.py defaults.",
        "# Edit this file to override defaults without changing Python source.\n",
        "[stopwords]",
        f"english = {_arr(_DEFAULT_STOPWORDS_EN)}",
        f"names   = {_arr(_DEFAULT_NAME_STOPWORDS)}\n",
        "[markers]",
        f"english = {_arr(_DEFAULT_ENGLISH_MARKERS)}",
        f"french  = {_arr(_DEFAULT_FRENCH_MARKERS)}\n",
        "[known_terms]",
        _table(_DEFAULT_KNOWN_TERMS) + "\n",
        "[html]",
        f'block_tags         = {sorted(_DEFAULT_BLOCK_TAGS)}',
        f'skip_inside_tags   = {sorted(_DEFAULT_SKIP_INSIDE)}',
        f'inline_tags        = {sorted(_DEFAULT_INLINE_TAGS)}',
        f'non_story_filenames = {sorted(_DEFAULT_NON_STORY)}\n',
        "[number_words]",
        f"ones = {[_DEFAULT_ONES[i] for i in range(20)]}",
        f"tens = {[_DEFAULT_TENS[(i+2)*10] for i in range(8)]}\n",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {dest}")
