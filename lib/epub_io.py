"""EPUB I/O: spine ordering, HTML parsing, translation unit extraction."""
from __future__ import annotations
import re
from pathlib import Path

from lib.models import InlineMarker, TranslationUnit
from lib.constants import BLOCK_TAGS, SKIP_INSIDE_TAGS, INLINE_TAGS, NON_STORY_FILENAMES
from lib.char_graph import strip_inline_markers
from lib.text_utils import normalize_text, extract_chapter_number, requested_chapter_number, sanitize_keyboard_text

def item_name(item: object) -> str:
    return str(getattr(item, "file_name", "") or item.get_name())


def html_items_in_spine_order(book: object, epub_module: object) -> list[object]:
    html_items = [item for item in book.get_items() if isinstance(item, epub_module.EpubHtml)]
    by_id = {item.get_id(): item for item in html_items}
    ordered: list[object] = []
    seen: set[str] = set()

    for spine_entry in getattr(book, "spine", []):
        idref = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
        item = by_id.get(idref)
        if item is None:
            continue
        name = item_name(item)
        if name not in seen:
            ordered.append(item)
            seen.add(name)

    for item in html_items:
        name = item_name(item)
        if name not in seen:
            ordered.append(item)
            seen.add(name)

    return ordered


def parse_html(content: bytes):
    from bs4 import BeautifulSoup

    return BeautifulSoup(content, "xml")


def first_heading_text(soup: object) -> str:
    for tag_name in ("h1", "h2", "h3", "title"):
        tag = soup.find(tag_name)
        if tag:
            text = tag.get_text(" ", strip=True)
            if text:
                return text
    return ""


def is_non_story_item(item: object) -> bool:
    stem = normalize_text(Path(item_name(item)).stem)
    return any(marker in stem for marker in NON_STORY_FILENAMES) and "chapter" not in stem


def is_numbered_chapter(item: object) -> bool:
    if is_non_story_item(item):
        return False

    soup = parse_html(item.get_content())
    filename = normalize_text(Path(item_name(item)).stem)
    heading = first_heading_text(soup)
    return extract_chapter_number(filename) is not None or extract_chapter_number(heading) is not None


def score_chapter_candidate(item: object, chapter_number: int | None) -> int:
    if is_non_story_item(item):
        return -1000

    soup = parse_html(item.get_content())
    filename = Path(item_name(item)).stem
    heading = first_heading_text(soup)
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    item_id = str(item.get_id() or "")

    if chapter_number is None:
        return -1000

    score = 0
    if extract_chapter_number(filename) == chapter_number:
        score += 100
    if extract_chapter_number(item_id) == chapter_number:
        score += 95
    if extract_chapter_number(heading) == chapter_number:
        score += 90
    if extract_chapter_number(title) == chapter_number:
        score += 70
    return score


def list_xhtml_items(book: object, epub_module: object) -> None:
    for index, item in enumerate(html_items_in_spine_order(book, epub_module), 1):
        soup = parse_html(item.get_content())
        heading = first_heading_text(soup)
        suffix = f" | {heading}" if heading else ""
        print(sanitize_keyboard_text(f"{index:02d}: {item_name(item)}{suffix}"))


def select_chapter_items(
    book: object,
    epub_module: object,
    chapter: str,
    all_chapters: bool,
    xhtml_index: int | None,
) -> list[object]:
    items = html_items_in_spine_order(book, epub_module)

    if xhtml_index is not None:
        if xhtml_index < 1 or xhtml_index > len(items):
            raise ValueError(f"--xhtml-index must be between 1 and {len(items)}.")
        return [items[xhtml_index - 1]]

    if all_chapters:
        chapters = [item for item in items if is_numbered_chapter(item)]
        if not chapters:
            raise ValueError("No numbered chapter files were found in this EPUB.")
        return chapters

    chapter_number = requested_chapter_number(chapter)
    scored = [(score_chapter_candidate(item, chapter_number), item) for item in items]
    scored = [(score, item) for score, item in scored if score > 0]
    if not scored:
        available = ", ".join(item_name(item) for item in items[:20])
        raise ValueError(
            f"Could not find real Chapter {chapter!r}. First XHTML files seen: {available}"
        )

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [scored[0][1]]


def normalize_marked_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t\n]+", " ", value)
    value = re.sub(r"\s+(\[\[/?)", r" \1", value)
    value = re.sub(r"(\]\])\s+", r"\1 ", value)
    return value.strip()


def marked_text_from_tag(tag: object) -> tuple[str, list[InlineMarker]]:
    from bs4 import NavigableString

    markers: list[InlineMarker] = []

    def walk(node: object) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        name = getattr(node, "name", None)
        if name is None:
            return ""
        text = "".join(walk(child) for child in getattr(node, "contents", []))
        if name in INLINE_TAGS and text.strip():
            marker_id = f"ZX{name.upper()}{len(markers) + 1}X"
            markers.append(InlineMarker(marker_id=marker_id, tag_name=name, attrs=dict(node.attrs)))
            return f"[[{marker_id}]]{text}[[/{marker_id}]]"
        return text

    marked = "".join(walk(child) for child in getattr(tag, "contents", []))
    return normalize_marked_text(marked), markers


def extract_translation_units(soup: object) -> list[TranslationUnit]:
    units: list[TranslationUnit] = []

    for tag in soup.find_all(tuple(BLOCK_TAGS)):
        if tag.find(tuple(SKIP_INSIDE_TAGS)):
            continue
        if tag.find(tuple(BLOCK_TAGS)):
            continue

        text = tag.get_text(" ", strip=True)
        if should_translate(text):
            marked_text, markers = marked_text_from_tag(tag)
            units.append(
                TranslationUnit(
                    tag=tag,
                    text=text,
                    marked_text=marked_text or text,
                    markers=markers,
                )
            )

    return units


def is_empty_anchor(node: object) -> bool:
    return (
        getattr(node, "name", None) == "a"
        and not node.get_text(strip=True)
        and (node.has_attr("id") or node.has_attr("name"))
    )


def append_text_node(parent: object, text: str) -> None:
    from bs4 import NavigableString

    if text:
        parent.append(NavigableString(text))


def marker_nodes(translated_text: str, markers: list[InlineMarker]) -> list[object]:
    from bs4 import BeautifulSoup

    marker_by_id = {marker.marker_id: marker for marker in markers}
    scratch = BeautifulSoup("", "xml")
    root = scratch.new_tag("root")
    stack: list[tuple[str, object]] = [("__root__", root)]
    pattern = re.compile(r"\[\[(/?)(ZX[A-Z]+[0-9]+X)\]\]")
    cursor = 0

    for match in pattern.finditer(translated_text):
        append_text_node(stack[-1][1], translated_text[cursor:match.start()])
        closing, marker_id = match.groups()
        cursor = match.end()

        if marker_id not in marker_by_id:
            continue
        if closing:
            if len(stack) > 1 and stack[-1][0] == marker_id:
                stack.pop()
            continue

        marker = marker_by_id[marker_id]
        new_tag = scratch.new_tag(marker.tag_name, attrs=marker.attrs)
        stack[-1][1].append(new_tag)
        stack.append((marker_id, new_tag))

    append_text_node(stack[-1][1], translated_text[cursor:])
    return list(root.contents)


def replace_tag_text(tag: object, translated_text: str, markers: list[InlineMarker] | None = None) -> None:
    from bs4 import NavigableString

    preserved_prefix = []
    while tag.contents and is_empty_anchor(tag.contents[0]):
        preserved_prefix.append(tag.contents[0].extract())

    tag.clear()
    for node in preserved_prefix:
        tag.append(node)

    cleaned = translated_text.strip()
    if markers and not missing_markers(cleaned, markers):
        for node in marker_nodes(cleaned, markers):
            tag.append(node)
        return

    tag.append(NavigableString(strip_inline_markers(cleaned)))


def write_epub_overwrite(epub_module: object, output_path: Path, book: object) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"Overwriting existing output: {output_path}")
    epub_module.write_epub(str(output_path), book, {})
