"""Shared dataclasses for the translation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class InlineMarker:
    marker_id: str
    tag_name: str
    attrs: dict[str, object]


@dataclass
class TranslationUnit:
    tag: object
    text: str
    marked_text: str
    markers: list[InlineMarker]


@dataclass
class TranslatorHandle:
    name: str
    translate: Callable[[str], str]
    cleanup: Callable[[], None]
    preload: Callable[[], None]
    translate_many: Callable[[list[str], int], list[str]] | None = None
    translate_many_nbest: Callable[[list[str], int, int], list[list[str]]] | None = None
    count_tokens_many: Callable[[list[str]], list[int]] | None = None


@dataclass
class RunMode:
    output_name: str
    draft_model: str
    patcher: str
    direct_model: str | None
    literary: bool = False


@dataclass
class ChunkRecord:
    index: int
    unit: TranslationUnit
    source: str
    draft: str = ""
    final: str = ""
    flags: list[str] | None = None
    speaker: str = ""
    addressee: str = ""
    register_hint: str = ""
    mentioned_characters: list[str] | None = None
    scene_participants: list[str] | None = None
    style_hints: list[str] | None = None
    alt_drafts: dict[str, str] | None = None


@dataclass
class ChapterEntityGraph:
    chapter_name: str
    characters: list[str]
    aliases: dict[str, str]
    relationships: dict[str, str]
    style_rules: list[str]
    dialogue_edges: list[dict[str, object]]
    scene_participants_by_chunk: dict[int, list[str]]
    ambiguous_dialogue_chunks: list[int]


@dataclass
class RecordBatch:
    records: list[ChunkRecord]
    token_count: int


@dataclass
class TranslationBatchItem:
    records: list[ChunkRecord]
    parts: list[str]
    delimiters: list[str] | None = None


@dataclass
class CandidateChoice:
    model_name: str
    text: str
    score: float
    raw_score: float | None
    engine: str


@dataclass
class PatchDecision:
    record: ChunkRecord
    needs_patch: bool
    reasons: list[str]
    raw_score: float | None = None
    heuristic_score: float | None = None


@dataclass
class ChapterRunResult:
    chunks: int
    validation_failures: int
    evaluation: dict[str, object]
