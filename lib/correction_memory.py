"""Accepted correction memory for literary post-editing."""
from __future__ import annotations
import json, re
from pathlib import Path
from lib.models import ChunkRecord
from lib.logging_utils import write_progress
from lib.registries import CORRECTION_MEMORY_VERSION, DEFAULT_CORRECTION_MEMORY_DIR
from lib.text_utils import normalize_text

def correction_memory_path(args: argparse.Namespace) -> Path:
    return Path(args.correction_memory_dir) / "literary_corrections.jsonl"


def normalize_memory_text(value: str) -> str:
    value = strip_inline_markers(value).lower()
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[^a-z0-9\u00c0-\u00ff']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def content_token_set(value: str) -> set[str]:
    return {
        token for token in word_tokens(normalize_memory_text(value))
        if len(token) > 2 and token not in TRANSLATION_STOPWORDS
    }


def token_jaccard(left: str, right: str) -> float:
    left_tokens = content_token_set(left)
    right_tokens = content_token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def correction_entry_key(entry: dict[str, object]) -> str:
    payload = {
        "source": entry.get("source", ""),
        "draft": entry.get("draft", ""),
        "final": entry.get("final", ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_correction_memory(args: argparse.Namespace) -> list[dict[str, object]]:
    if not getattr(args, "correction_memory", False):
        return []
    cached = getattr(load_correction_memory, "_cache", None)
    cached_path = getattr(load_correction_memory, "_path", None)
    path = correction_memory_path(args)
    if cached is not None and cached_path == str(path):
        return cached
    if not path.is_file():
        setattr(load_correction_memory, "_cache", [])
        setattr(load_correction_memory, "_path", str(path))
        return []

    entries: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict) and item.get("source") and item.get("final"):
                    entries.append(item)
    except Exception as exc:
        log_error(args, "Correction memory read failed", exc, {"path": path})
        entries = []

    max_entries = max(1, int(getattr(args, "correction_memory_max_entries", 5000)))
    entries = entries[-max_entries:]
    setattr(load_correction_memory, "_cache", entries)
    setattr(load_correction_memory, "_path", str(path))
    return entries


def correction_memory_similarity(record: ChunkRecord, entry: dict[str, object]) -> float:
    source_score = token_jaccard(record.source, str(entry.get("source", "")))
    draft_score = token_jaccard(record.draft, str(entry.get("draft", ""))) if record.draft else 0.0
    register_bonus = 0.05 if record.register_hint and record.register_hint == entry.get("register_hint") else 0.0
    return min(1.0, max(source_score, 0.65 * source_score + 0.35 * draft_score) + register_bonus)


def _faiss_correction_memory_search(
    record: ChunkRecord,
    entries: list[dict[str, object]],
    top_k: int,
    threshold: float,
) -> list[tuple[float, dict[str, object]]] | None:
    """Return FAISS-ranked results, or None if faiss/sentence-transformers unavailable."""
    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    cache_key = id(entries)
    cache = getattr(_faiss_correction_memory_search, "_cache", {})
    if cache.get("key") != cache_key or len(cache.get("entries", [])) != len(entries):
        encoder = cache.get("encoder") or SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        sources = [str(e.get("source", "")) for e in entries]
        vecs = encoder.encode(
            sources, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        vecs = np.array(vecs, dtype="float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        _faiss_correction_memory_search._cache = {
            "key": cache_key,
            "entries": entries,
            "index": index,
            "encoder": encoder,
        }
        cache = _faiss_correction_memory_search._cache

    encoder = cache["encoder"]
    faiss_index = cache["index"]
    query = encoder.encode(
        [record.source], batch_size=1, show_progress_bar=False, normalize_embeddings=True
    )
    query = np.array(query, dtype="float32")
    distances, indices = faiss_index.search(query, min(top_k * 4, len(entries)))
    results = [
        (float(score), entries[idx])
        for score, idx in zip(distances[0], indices[0])
        if idx >= 0 and float(score) >= threshold
    ]
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def similar_correction_memory_entries(
    args: argparse.Namespace,
    record: ChunkRecord,
) -> list[tuple[float, dict[str, object]]]:
    if not getattr(args, "correction_memory", False):
        return []
    threshold = max(0.0, float(getattr(args, "correction_memory_threshold", 0.56)))
    top_k = max(0, int(getattr(args, "correction_memory_examples", 3)))
    entries = load_correction_memory(args)

    faiss_results = _faiss_correction_memory_search(record, entries, top_k, threshold)
    if faiss_results is not None:
        return faiss_results

    ranked = [
        (correction_memory_similarity(record, entry), entry)
        for entry in entries
    ]
    ranked = [item for item in ranked if item[0] >= threshold]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:top_k]


def correction_memory_examples_text(args: argparse.Namespace, record: ChunkRecord) -> str:
    examples = similar_correction_memory_entries(args, record)
    if not examples:
        return ""
    blocks = []
    for index, (score, entry) in enumerate(examples, 1):
        fixed = entry.get("flags_fixed", [])
        fixed_text = ", ".join(str(item) for item in fixed) if isinstance(fixed, list) else str(fixed)
        blocks.append(
            f"Example {index} similarity {score:.2f}\n"
            f"Source: {entry.get('source', '')}\n"
            f"Draft: {entry.get('draft', '')}\n"
            f"Accepted French: {entry.get('final', '')}\n"
            f"Fixed: {fixed_text}"
        )
    return "\n\n".join(blocks)


def exact_correction_memory_candidate(args: argparse.Namespace, record: ChunkRecord) -> str:
    if not getattr(args, "correction_memory", False):
        return ""
    normalized_source = normalize_memory_text(record.source)
    normalized_draft = normalize_memory_text(record.draft)
    for entry in reversed(load_correction_memory(args)):
        if normalize_memory_text(str(entry.get("source", ""))) != normalized_source:
            continue
        entry_draft = normalize_memory_text(str(entry.get("draft", "")))
        if entry_draft and normalized_draft and token_jaccard(entry_draft, normalized_draft) < 0.85:
            continue
        candidate = str(entry.get("final", "")).strip()
        if candidate and not validate_record_translation(record, record.draft, candidate):
            return candidate
    return ""


def save_correction_memory(
    args: argparse.Namespace,
    records: list[ChunkRecord],
    chapter_name: str,
    patcher: str,
) -> None:
    if not getattr(args, "correction_memory", False) or args.dry_run:
        return

    path = correction_memory_path(args)
    existing_keys = {
        correction_entry_key(entry)
        for entry in load_correction_memory(args)
    }
    new_entries: list[dict[str, object]] = []
    for record in records:
        final = record.final or record.draft
        if not final or final == record.draft:
            continue
        final_flags = validate_record_translation(record, record.draft, final)
        if final_flags:
            continue
        draft_flags = record.flags or validate_record_translation(record, record.draft, record.draft)
        entry = {
            "version": CORRECTION_MEMORY_VERSION,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "chapter": chapter_name,
            "patcher": patcher,
            "source": record.source,
            "draft": record.draft,
            "final": final,
            "alt_drafts": record.alt_drafts or {},
            "flags_fixed": draft_flags,
            "speaker": record.speaker,
            "addressee": record.addressee,
            "register_hint": record.register_hint,
            "mentioned_characters": record.mentioned_characters or [],
            "scene_participants": record.scene_participants or [],
            "style_hints": record.style_hints or [],
        }
        key = correction_entry_key(entry)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_entries.append(entry)

    if not new_entries:
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for entry in new_entries:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        cached = load_correction_memory(args)
        cached.extend(new_entries)
        max_entries = max(1, int(getattr(args, "correction_memory_max_entries", 5000)))
        setattr(load_correction_memory, "_cache", cached[-max_entries:])
        write_progress(f"Correction memory: stored {len(new_entries)} accepted edit(s).")
    except Exception as exc:
        log_error(args, "Correction memory write failed", exc, {"path": path})
