from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString
import copy
from translatorNLLB import *


# way to handle italics
# translation lack of sense  due to lack of context
# ex i feel *dirty* is translated as i feel + i m dirty
def remove_pasta(soup):
    for tag in soup.find_all(["i", "em"]):
        tag.unwrap()


def croutons(soupe):
    for child in soupe.children:
        if isinstance(child, NavigableString):
            # Translate text nodes
            text = str(child).strip()
            if text:
                translated = largetranslate(text)
                child.replace_with(translated)
        else:  # recursive croutons !
            croutons(child)


def cook(filename):
    if filename.endswith(".epub"):
        base = filename[:-5]
    else:
        base = filename
    book = epub.read_epub(filename)
    # Get all HTML chapters
    html_items = [item for item in book.get_items() if isinstance(item, epub.EpubHtml)]
    for item in tqdm(
        html_items, desc="Translating chapters", unit="chapter", leave=False
    ):
        # Parse HTML content
        soup = BeautifulSoup(item.content, "html.parser")
        # translate the soup
        remove_pasta(soup)
        croutons(soup)
        preview = soup.get_text()[:300].replace("\n", " ")
        item.content = str(soup).encode("utf-8", errors="replace")
        tqdm.write(f"✓ {item.get_id()}: {preview}...")
    epub.write_epub(f"{base}.translated.epub", book)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [epub files]")
        print("   or: python script.py *.epub")
        sys.exit(1)
    epub_files = sys.argv[1:]
    print(f"Found {len(epub_files)} file(s) to translate")
    print("It will take a little while. Feel free to go out and get some fresh air.")

    for epub_file in tqdm(epub_files, desc="Books", unit="book"):
        try:
            output_file = translate_epub(epub_file)
            tqdm.write(f"✓ {epub_file} → {output_file}")
        except Exception as e:
            tqdm.write(f"✗ Error processing {epub_file}: {e}")
            continue
    print(f"All done! Translated {len(epub_files)} file(s)")
