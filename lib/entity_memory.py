"""Named entity consistency tracking across chapters.

Extracts named entities from source text via spaCy, looks up previously seen
entitytranslation pairs, and surfaces conflicts in the final output.

Cache file: cache/entity_memory/{book_slug}.json
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

DEFAULT_ENTITY_MEMORY_DIR = Path("cache") / "entity_memory"


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9_-]", "_", text.lower()).strip("_")[:80]


def _load_spacy():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    except ImportError as exc:
        raise RuntimeError("spacy is required for entity memory. Run: pip install spacy") from exc


class EntityMemory:
    """Per-book entityFrench translation memory."""

    def __init__(self, book_path: str | Path, memory_dir: str | Path = DEFAULT_ENTITY_MEMORY_DIR):
        book_path = Path(book_path)
        self._path = Path(memory_dir) / f"{_slug(book_path.stem)}.json"
        self._data: dict[str, dict[str, int]] = {}  # entity -> {fr_form -> count}
        self._nlp = None
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def _save_to_disk(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _nlp_model(self):
        if self._nlp is None:
            self._nlp = _load_spacy()
        return self._nlp

    def extract_entities(self, source_text: str) -> list[str]:
        """Return list of named entity strings (PERSON, ORG, LOC, PRODUCT) from source."""
        nlp = self._nlp_model()
        doc = nlp(source_text[:100_000])
        return list({
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG", "LOC", "FAC", "PRODUCT", "EVENT"}
            and len(ent.text.strip()) > 1
        })

    def update(self, entity: str, fr_form: str) -> None:
        """Record that `entity` was translated as `fr_form` in this chapter."""
        key = entity.lower().strip()
        if not key or not fr_form.strip():
            return
        counts = self._data.setdefault(key, {})
        counts[fr_form.strip()] = counts.get(fr_form.strip(), 0) + 1

    def lookup(self, entity: str) -> str | None:
        """Return the most-used French form for this entity, or None if unseen."""
        key = entity.lower().strip()
        counts = self._data.get(key)
        if not counts:
            return None
        return max(counts, key=counts.__getitem__)

    def find_entity_in_translation(self, entity: str, translation: str) -> str | None:
        """Heuristic: find entity or a likely French form in the translated text."""
        # Simple: entity itself might be kept (names usually are)
        if entity in translation:
            return entity
        return None

    def update_from_chapter(self, entities: list[str], source_chunks: list[str], translated_chunks: list[str]) -> None:
        """Update memory by finding entities in chunk pairs and recording translations."""
        for entity in entities:
            for src, tgt in zip(source_chunks, translated_chunks):
                if entity.lower() not in src.lower():
                    continue
                found = self.find_entity_in_translation(entity, tgt)
                if found:
                    self.update(entity, found)
        self._save_to_disk()

    def consistency_warnings(self, entities: list[str]) -> list[str]:
        """Return warnings for entities that have been translated in multiple ways."""
        warnings = []
        for entity in entities:
            key = entity.lower().strip()
            counts = self._data.get(key, {})
            if len(counts) > 1:
                forms = sorted(counts, key=counts.__getitem__, reverse=True)
                warnings.append(
                    f"Entity '{entity}' translated multiple ways: "
                    + ", ".join(f'"{f}" ({counts[f]}x)' for f in forms[:3])
                )
        return warnings
