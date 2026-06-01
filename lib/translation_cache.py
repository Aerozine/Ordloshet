"""Translation and pipeline result caching."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from lib.models import ChunkRecord
from lib.registries import TRANSLATION_CACHE_VERSION, PIPELINE_CACHE_VERSION

def translation_cache_enabled(args: argparse.Namespace) -> bool:
    return bool(args.cache_translations and not args.dry_run)


def translation_cache_payload(
    records: list[ChunkRecord],
    handle: TranslatorHandle,
    args: argparse.Namespace,
) -> dict[str, object]:
    sources = [
        {
            "index": record.index,
            "source": record.source,
            "markers": [
                {
                    "id": marker.marker_id,
                    "tag": marker.tag_name,
                    "attrs": marker.attrs,
                }
                for marker in record.unit.markers
            ],
        }
        for record in records
    ]
    return {
        "version": TRANSLATION_CACHE_VERSION,
        "model": handle.name,
        "french_typography": bool(args.french_typography),
        "sources": sources,
    }


def translation_cache_key(
    records: list[ChunkRecord],
    handle: TranslatorHandle,
    args: argparse.Namespace,
) -> str:
    payload = translation_cache_payload(records, handle, args)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def translation_cache_path(
    records: list[ChunkRecord],
    handle: TranslatorHandle,
    args: argparse.Namespace,
) -> Path:
    key = translation_cache_key(records, handle, args)
    model_dir = safe_output_stem(handle.name.replace("/", "_"))
    return Path(args.translation_cache_dir) / model_dir / f"{key}.json"


def read_translation_cache(
    records: list[ChunkRecord],
    handle: TranslatorHandle,
    args: argparse.Namespace,
) -> list[str] | None:
    if not translation_cache_enabled(args) or args.refresh_translation_cache:
        return None

    path = translation_cache_path(records, handle, args)
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
        outputs = payload.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(records):
            return None
        if not all(isinstance(output, str) for output in outputs):
            return None
        return outputs
    except Exception as exc:
        log_error(
            args,
            "Translation cache read failed",
            exc,
            {"model": handle.name, "cache_path": path},
        )
        return None


def write_translation_cache(
    records: list[ChunkRecord],
    handle: TranslatorHandle,
    args: argparse.Namespace,
    outputs: list[str],
) -> None:
    if not translation_cache_enabled(args) or len(outputs) != len(records):
        return

    path = translation_cache_path(records, handle, args)
    payload = translation_cache_payload(records, handle, args)
    payload["outputs"] = outputs
    payload["created_at"] = dt.datetime.now().isoformat(timespec="seconds")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file, ensure_ascii=False, sort_keys=True)
            cache_file.write("\n")
        temp_path.replace(path)
    except Exception as exc:
        log_error(
            args,
            "Translation cache write failed",
            exc,
            {"model": handle.name, "cache_path": path},
        )


def _chapter_checkpoint_path(
    args: argparse.Namespace, chapter_name: str, model_name: str
) -> Path | None:
    if not getattr(args, "chapter_checkpoint", False) or args.dry_run:
        return None
    book_stem = safe_output_stem(getattr(args, "_current_book_stem", "book") or "book")
    chapter_slug = safe_output_stem(chapter_name)[:40]
    model_slug = safe_output_stem(model_name)
    return Path("cache") / "progress" / book_stem / f"{chapter_slug}_{model_slug}.jsonl"


def _read_chapter_checkpoint(path: Path | None) -> dict[int, str]:
    if path is None or not path.is_file():
        return {}
    results: dict[int, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            results[int(item["index"])] = str(item["translation"])
    except Exception:
        pass
    return results


def _append_chapter_checkpoint(path: Path | None, index: int, translation: str) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"index": index, "translation": translation}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _clear_chapter_checkpoint(path: Path | None) -> None:
    if path is not None and path.is_file():
        try:
            path.unlink()
        except Exception:
            pass


def pipeline_cache_enabled(args: argparse.Namespace) -> bool:
    return bool(args.cache_translations and not args.dry_run)


def pipeline_cache_path(args: argparse.Namespace, namespace: str, payload: dict[str, object]) -> Path:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    key = hashlib.sha256(raw).hexdigest()
    return Path(args.translation_cache_dir) / "_pipeline" / namespace / f"{key}.json"


def read_pipeline_cache(
    args: argparse.Namespace,
    namespace: str,
    payload: dict[str, object],
) -> object | None:
    if not pipeline_cache_enabled(args) or args.refresh_translation_cache:
        return None

    path = pipeline_cache_path(args, namespace, payload)
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
        return cached.get("value")
    except Exception as exc:
        log_error(
            args,
            "Pipeline cache read failed",
            exc,
            {"namespace": namespace, "cache_path": path},
        )
        return None


def write_pipeline_cache(
    args: argparse.Namespace,
    namespace: str,
    payload: dict[str, object],
    value: object,
) -> None:
    if not pipeline_cache_enabled(args):
        return

    path = pipeline_cache_path(args, namespace, payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as cache_file:
            json.dump(
                {
                    "version": PIPELINE_CACHE_VERSION,
                    "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "value": value,
                },
                cache_file,
                ensure_ascii=False,
                sort_keys=True,
            )
            cache_file.write("\n")
        temp_path.replace(path)
    except Exception as exc:
        log_error(
            args,
            "Pipeline cache write failed",
            exc,
            {"namespace": namespace, "cache_path": path},
        )


def cached_text_call(
    args: argparse.Namespace,
    namespace: str,
    payload: dict[str, object],
    fn: Callable[[], str],
) -> str:
    full_payload = {
        "version": PIPELINE_CACHE_VERSION,
        "namespace": namespace,
        **payload,
    }
    cached = read_pipeline_cache(args, namespace, full_payload)
    if isinstance(cached, str):
        return cached
    value = fn()
    write_pipeline_cache(args, namespace, full_payload, value)
    return value
