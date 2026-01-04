from ebooklib import epub
from bs4 import BeautifulSoup
from translator import translate

# Load the EPUB file
book = epub.read_epub('file.epub')
for item in book.get_items():
    print(item) 
# get all chapters in xhtml format 
#do a parser function 

for item in book.get_items():
    # Check if the item is an HTML document (EpubHtml)
    if isinstance(item, epub.EpubHtml):
        content = item.content
        print(content) 
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        # Extract and print the text content of the chapter
        text = soup.get_text()
        print(f"Content of {item.get_id()}:\n")
        print(text[:500])  # Print the first 500 characters as a preview
        print("\n" + "="*50 + "\n")  # Separator between chapters
        For every spoon of soup , translate and replace on a cloned structures
        """
# the bee script
print(translate("According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible."))
