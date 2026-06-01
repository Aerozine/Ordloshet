"""TOML book configuration loader.

Loads per-book character aliases, genders, relationships, and glossary rules,
then patches lib.char_graph globals directly (no epub module reference needed).

TOML format (generate a template with: python -c "from lib.book_config import print_template; print_template()"):

    [characters.aliases]
    "Surname" = "Firstname"

    [characters.genders]
    "Name" = "male"

    [characters.relationships]
    "Name1|Name2" = "intimate_partner"

    [glossary.term_label]
    source = ["\\bterm\\b"]
    preferred = ["\\bfrench equivalent\\b"]
    forbidden = []
"""
from __future__ import annotations

import sys
from pathlib import Path


def load_book_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Book config not found: {path}")

    if sys.version_info >= (3, 11):
        import tomllib
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                "tomli is required on Python < 3.11. Run: pip install tomli"
            ) from exc
        data = tomllib.loads(path.read_text(encoding="utf-8"))

    return data


def apply_book_config(config_path: str | Path, epub_module=None) -> None:
    """Patch character globals from a TOML book config file."""
    data = load_book_config(config_path)
    apply_book_config_dict(data, epub_module)


def apply_book_config_dict(data: dict, epub_module=None) -> None:
    """Patch character globals from an already-loaded config dict.

    Patches lib.char_graph globals directly so that all functions using those
    globals (including ones imported into other modules) see the update.
    The ``epub_module`` parameter is accepted for backward compatibility but
    is no longer required.
    """
    from lib.char_graph import patch_character_data

    chars = data.get("characters", {})
    aliases = chars.get("aliases", {})
    genders = chars.get("genders", {})
    relationships = chars.get("relationships", {})
    glossary = data.get("glossary", {})
    force_tu = bool(data.get("register", {}).get("force_tu", False))

    patch_character_data(
        aliases=aliases or None,
        genders=genders or None,
        relationships=relationships or None,
        glossary=glossary or None,
        force_tu=force_tu,
    )

    print(
        f"Book config applied: "
        f"{len(aliases)} aliases, {len(genders)} genders, "
        f"{len(relationships)} relationships, {len(glossary)} glossary rules"
        f"{', force_tu=true' if force_tu else ''}."
    )


_TEMPLATE = """\
# Manual book configuration — overrides auto-analysis for this title.
# Pass with: python epub.py mybook.epub --book-config mybook.toml
# All sections are optional.

[characters.aliases]
# "Surname" = "Firstname"

[characters.genders]
# "Name" = "male"  # or "female" | "neutral"

[characters.relationships]
# "Name1|Name2" = "intimate_partner"  # or "friends" | "rivals" | "family"

[register]
# force_tu = true

# [glossary.term_label]
# source    = ["\\\\bterm\\\\b"]
# preferred = ["\\\\bfrench equivalent\\\\b"]
# forbidden = []
"""


def print_template() -> None:
    """Print a TOML template to stdout."""
    print(_TEMPLATE)
