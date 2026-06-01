"""Progress display and error logging utilities."""
from __future__ import annotations
import json, os, sys, traceback
from pathlib import Path

try:
    from tqdm.std import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        @staticmethod
        def write(msg: str, **_: object) -> None:
            print(msg)

from lib.text_utils import sanitize_keyboard_text, progress_text

def error_log_path(args: argparse.Namespace) -> Path | None:
    if args.no_error_log:
        return None
    return Path(args.error_log)


def init_error_log(args: argparse.Namespace) -> None:
    path = error_log_path(args)
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    books = ", ".join(args.book)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 80 + "\n")
        handle.write(f"Run started: {timestamp}\n")
        handle.write(f"Command: {' '.join(sys.argv)}\n")
        handle.write(f"Model: {args.model}\n")
        handle.write(f"Strategy: {args.strategy}\n")
        handle.write(f"Book(s): {books}\n")
        handle.write(f"Chapter: {args.chapter}\n")
        if args.xhtml_index is not None:
            handle.write(f"XHTML index: {args.xhtml_index}\n")
        handle.write("=" * 80 + "\n")


def mark_logged(exc: BaseException) -> None:
    try:
        setattr(exc, "_ordloshet_logged", True)
    except Exception:
        pass


def was_logged(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if getattr(current, "_ordloshet_logged", False):
            return True
        current = current.__cause__ or current.__context__
    return False


def log_error(
    args: argparse.Namespace,
    title: str,
    exc: BaseException | None = None,
    details: dict[str, object] | None = None,
) -> None:
    path = error_log_path(args)
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "-" * 80 + "\n")
        handle.write(f"Time: {timestamp}\n")
        handle.write(f"Error: {log_text(title)}\n")
        if details:
            for key, value in details.items():
                handle.write(f"{log_text(key)}: {log_text(value)}\n")
        if exc is not None:
            handle.write("Traceback:\n")
            handle.write(
                log_text(
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                )
            )
            handle.write("\n")
            mark_logged(exc)


def print_error_log_location(args: argparse.Namespace) -> None:
    path = error_log_path(args)
    if path is not None:
        print(f"Full error log: {path}")


def progress_position(value: int) -> int | None:
    return value if sys.stdout.isatty() else 0


def progress_kwargs(position: int) -> dict[str, object]:
    return {
        "ascii": True,
        "dynamic_ncols": False,
        "file": sys.stdout,
        "leave": True,
        "mininterval": 0.5,
        "ncols": PROGRESS_WIDTH,
        "position": progress_position(position),
    }


def refresh_progress(progress: object) -> None:
    refresh = getattr(progress, "refresh", None)
    if callable(refresh):
        refresh()


def pipeline_step(progress: object | None, label: str, amount: int = 1) -> None:
    if progress is None:
        return
    try:
        progress.set_postfix_str(label, refresh=False)
    except TypeError:
        progress.set_postfix_str(label)
    progress.update(amount)


def set_pipeline_total(progress: object | None, total: int) -> None:
    if progress is None:
        return
    try:
        progress.total = total
        refresh_progress(progress)
    except Exception:
        pass


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


def write_progress(message: str) -> None:
    write = getattr(tqdm, "write", None)
    try:
        if callable(write):
            write(message, file=sys.stdout)
        else:
            print(message)
    except Exception as exc:
        if is_broken_output_error(exc):
            silence_broken_console()
            return
        raise


def print_chunk_text(
    chapter_name: str,
    index: int,
    total: int,
    label: str,
    text: str,
    max_chars: int,
) -> None:
    write_progress(
        f"\n[{chapter_name}] chunk {index}/{total} {label}:\n"
        f"{progress_text(text, max_chars)}"
    )
