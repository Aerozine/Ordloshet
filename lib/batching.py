"""Translation batching and segment grouping."""
from __future__ import annotations
import re
from lib.models import ChunkRecord, TranslationUnit, RecordBatch, TranslationBatchItem
from lib.char_graph import strip_inline_markers
from lib.text_utils import word_tokens
from lib.constants import TRANSLATION_STOPWORDS

def estimate_token_count(text: str) -> int:
    if not text:
        return 1
    word_count = len(re.findall(r"\S+", text))
    char_count = len(text)
    return max(1, word_count, (char_count + 3) // 4)


def token_counts_for_records(
    records: list[ChunkRecord],
    handle: TranslatorHandle,
    args: argparse.Namespace,
) -> list[int]:
    texts = [record.source for record in records]
    if not args.dry_run and handle.count_tokens_many is not None:
        try:
            counts = handle.count_tokens_many(texts)
            if len(counts) == len(records) and all(count > 0 for count in counts):
                return counts
        except Exception as exc:
            log_error(
                args,
                "Tokenizer token counting failed; rough estimates used",
                exc,
                {"model": handle.name},
            )
    return [estimate_token_count(text) for text in texts]


def make_record_batches(
    records: list[ChunkRecord],
    handle: TranslatorHandle,
    args: argparse.Namespace,
) -> list[RecordBatch]:
    if not records:
        return []

    supports_batching = handle.translate_many is not None
    max_batch_size = max(1, args.batch_size if supports_batching else 1)
    token_limit = max(0, args.batch_token_limit)

    # Sort by source length so padding waste is minimised in each batch.
    # Build an index map so we can restore original order after translation.
    sorted_records = sorted(records, key=lambda r: len(r.source))

    if not supports_batching or not args.dynamic_batching or token_limit <= 0:
        batches = []
        for start in range(0, len(sorted_records), max_batch_size):
            batch_records = sorted_records[start:start + max_batch_size]
            token_count = sum(estimate_token_count(record.source) for record in batch_records)
            batches.append(RecordBatch(records=batch_records, token_count=token_count))
        return batches

    token_counts = token_counts_for_records(sorted_records, handle, args)
    batches: list[RecordBatch] = []
    current_records: list[ChunkRecord] = []
    current_tokens = 0

    for record, token_count in zip(sorted_records, token_counts):
        token_count = max(1, token_count)
        would_overflow_tokens = current_records and current_tokens + token_count > token_limit
        would_overflow_size = len(current_records) >= max_batch_size
        if would_overflow_tokens or would_overflow_size:
            batches.append(RecordBatch(records=current_records, token_count=current_tokens))
            current_records = []
            current_tokens = 0
        current_records.append(record)
        current_tokens += token_count

    if current_records:
        batches.append(RecordBatch(records=current_records, token_count=current_tokens))

    return batches


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?»\'"])\s+(?=[A-Z"«—\-])')
_GROUP_SEGMENT_RE = re.compile(r"\[\[ZXSEG\d+X\]\]|ZXSEG\d+X")


def _split_long_source(source: str, max_words: int = 350) -> list[str]:
    """Split source at sentence boundary if it exceeds max_words. Returns [source] if short enough."""
    words = source.split()
    if len(words) <= max_words:
        return [source]
    sentences = _SENT_SPLIT_RE.split(source)
    if len(sentences) <= 1:
        return [source]
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for sent in sentences:
        n = len(sent.split())
        if count + n > max_words and current:
            parts.append(" ".join(current))
            current = [sent]
            count = n
        else:
            current.append(sent)
            count += n
    if current:
        parts.append(" ".join(current))
    return parts if len(parts) > 1 else [source]


def _short_chunk_groupable(record: ChunkRecord, args: argparse.Namespace) -> bool:
    if not getattr(args, "group_short_chunks", True):
        return False
    source = strip_inline_markers(record.source).strip()
    if not source:
        return False
    if len(source) > max(1, int(getattr(args, "short_chunk_chars", 60))):
        return False
    return len(record.source.split()) <= 12


def _make_translation_batch_items(
    batch: list[ChunkRecord],
    args: argparse.Namespace,
) -> list[TranslationBatchItem]:
    items: list[TranslationBatchItem] = []
    i = 0
    while i < len(batch):
        record = batch[i]
        if _short_chunk_groupable(record, args):
            group = [record]
            j = i + 1
            while j < len(batch) and len(group) < 4 and _short_chunk_groupable(batch[j], args):
                group.append(batch[j])
                j += 1
            if len(group) > 1:
                delimiters = [f"[[ZXSEG{k}X]]" for k in range(1, len(group))]
                grouped_source = group[0].source
                for delimiter, next_record in zip(delimiters, group[1:]):
                    grouped_source += f"\n{delimiter}\n{next_record.source}"
                items.append(TranslationBatchItem(records=group, parts=[grouped_source], delimiters=delimiters))
                i = j
                continue
        items.append(TranslationBatchItem(records=[record], parts=_split_long_source(record.source)))
        i += 1
    return items


def _split_grouped_translation(
    translated: str,
    delimiters: list[str],
    expected_count: int,
) -> list[str] | None:
    if not delimiters:
        return None
    parts = [part.strip() for part in _GROUP_SEGMENT_RE.split(translated) if part.strip()]
    if len(parts) == expected_count:
        return parts
    lines = [
        line.strip()
        for line in translated.splitlines()
        if line.strip() and not _GROUP_SEGMENT_RE.fullmatch(line.strip())
    ]
    if len(lines) == expected_count:
        return lines
    return None


def _collapse_translated_items(
    items: list[TranslationBatchItem],
    flat_translated: list[str],
    handle: TranslatorHandle,
    args: argparse.Namespace,
) -> list[str]:
    translated_batch: list[str] = []
    offset = 0
    for item in items:
        item_outputs = flat_translated[offset:offset + len(item.parts)]
        offset += len(item.parts)
        if len(item.records) == 1:
            translated_batch.append(" ".join(part.strip() for part in item_outputs if part.strip()))
            continue
        grouped_output = item_outputs[0] if item_outputs else ""
        split_output = _split_grouped_translation(
            grouped_output,
            item.delimiters or [],
            len(item.records),
        )
        if split_output is None:
            split_output = translate_batch(
                handle,
                [record.source for record in item.records],
                args,
                records=item.records,
            )
        translated_batch.extend(split_output)
    return translated_batch


def translate_batch(
    handle: TranslatorHandle,
    texts: list[str],
    args: argparse.Namespace,
    records: "list[ChunkRecord] | None" = None,
) -> list[str]:
    if args.dry_run:
        return texts
    n_best = getattr(args, "draft_n_best", 4)
    if handle.translate_many_nbest is not None and n_best > 1 and len(texts) > 0:
        all_hyps = handle.translate_many_nbest(texts, max(1, len(texts)), n_best)
        results = []
        for i, (text, hyps) in enumerate(zip(texts, all_hyps)):
            if len(hyps) <= 1:
                results.append(hyps[0] if hyps else text)
                continue
            rec = records[i] if records else None
            best = _pick_best_hypothesis(text, hyps, rec)
            results.append(best)
        return results
    if handle.translate_many is not None and len(texts) > 1:
        return handle.translate_many(texts, max(1, len(texts)))
    return [handle.translate(text) for text in texts]


def _pick_best_hypothesis(source: str, hyps: list[str], record: "ChunkRecord | None") -> str:
    """Pick the best hypothesis: prefer ones containing source names, then highest heuristic score."""
    if not hyps:
        return source
    # Names that must appear in the output
    required_names = source_names(source)
    # Filter to hypotheses that contain all required names (if any)
    if required_names:
        name_complete = [
            h for h in hyps
            if all(re.search(rf"\b{re.escape(n)}\b", h) for n in required_names)
        ]
        if name_complete:
            hyps = name_complete
    if len(hyps) == 1:
        return hyps[0]
    # Score remaining candidates with heuristic
    if record is not None:
        scored = [
            (candidate_score_for_record(record, h, source), h)
            for h in hyps
        ]
    else:
        scored = [
            (score_from_flags(h, source, validate_translation(source, source, h, [])), h)
            for h in hyps
        ]
    return max(scored, key=lambda x: x[0])[1]
