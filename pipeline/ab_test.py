"""pipeline/ab_test.py  -  Side-by-side translation comparison (A/B testing).

Creates an EPUB that displays multiple translation versions side-by-side for manual comparison.
Enables users to review different model outputs before selecting final version.
"""
import epub
from ebooklib import epub
from bs4 import BeautifulSoup


def compare_translations(book1_path: str, book2_path: str, 
                         output_path: str, title: str = None) -> str:
    """Create side-by-side comparison EPUB.
    
    Args:
        book1_path: First translation EPUB
        book2_path: Second translation EPUB  
        output_path: Output comparison EPUB path
        title: Optional custom title
    
    Returns:
        Output file path
    """
    # Read both source books
    book1 = epub.read_epub(book1_path)
    book2 = epub.read_epub(book2_path)
    
    # Create new book with metadata showing it's a comparison
    translated_book = epub.EpubBook()
    
    if title:
        translated_book.set_title(f"Comparison: {title}")
    else:
        # Extract common book name from paths
        import os
        base_name = os.path.splitext(os.path.basename(book1_path))[0]
        translated_book.set_title(f"A/B Test Comparison ({base_name})")
    
    translated_book.set_language('fr')
    translated_book.add_item(epub.EpubNcx())
    translated_book.add_item(epub.EpubToc())
    translated_book.add_item(epub.EpubNav())
    
    # Create stylesheets for side-by-side comparison
    style = """
    <style type="text/css">
        body { font-family: Georgia, serif; }
        .comparison-page { 
            display: flex; 
            height: 100vh; 
            overflow-y: auto;
        }
        .translation-column { 
            width: 50%; 
            padding: 20px; 
            overflow-y: auto;
            border-right: 2px solid #ccc;
            min-width: 400px;
        }
        .translation-column:last-child { border-right: none; }
        .original-text { color: #666; font-style: italic; font-size: 0.9em; margin-bottom: 1em; }
        .nllb-label { background: #e3f2fd; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
        .madlad-label { background: #fff3e0; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
        .opus-label { background: #f3e5f5; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
        .gemma-label { background: #e8f5e9; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
        .mistral-label { background: #fce4ec; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
    </style>
    """
    
    translated_book.add_item(epub.EpubCss(style_content=style))
    
    for item in book1.get_items():
        if isinstance(item, epub.EpubItem):
            name = item.get_name()
            
            # Copy non-HTML items (cover images, stylesheets, NCX, TOC)
            if name.endswith('.png') or name.endswith('.jpg') or \
               name.endswith('.css') or 'ncx' in name.lower() or \
               'toc' in name.lower():
                translated_book.add_item(item)
    
    # Create comparison pages for each chapter pair
    chap1_items = [i for i in book1.get_items() if isinstance(i, epub.EpubHtml)]
    chap2_items = [i for i in book2.get_items() if isinstance(i, epub.EpubHtml)]
    
    # Match chapters by filename
    matched_pairs = []
    for c1 in chap1_items:
        name1 = os.path.splitext(c1.get_name())[0]
        for c2 in chap2_items:
            name2 = os.path.splitext(c2.get_name())[0]
            if name1 == name2:
                matched_pairs.append((c1, c2))
    
    # Create comparison HTML for each pair
    from bs4 import BeautifulSoup
    
    soup1 = BeautifulSoup(chap1_items[0].content, 'html.parser')
    soup2 = BeautifulSoup(chap2_items[0].content, 'html.parser')
    
    # Extract chapter metadata  
    title_elem = None
    if soup1.find('title'):
        title_elem = soup1.find('title').get_text().strip() or \
                     os.path.splitext(c1.get_name())[0]
    
    comparison_html = f"""<div class="comparison-page">
<h1>Comparison: {title_elem}</h1>
<div class="translation-column nllb-label" id="left">
<h2>NLLB Translation</h2>
{str(soup1)}
</div>
<div class="translation-column madlad-label" id="right">
<h2>MADLAD Translation</h2>
{str(soup2)}
</div>
</div>"""
    
    comparison_item = epub.EpubItem(
        name=f"comparison_{os.path.splitext(c1.get_name())[0]}.xhtml",
        file_name=os.path.splitext(c1.get_name())[0].replace('.epub', '') + '.xhtml',
        content=comparison_html,
        media_type="application/xhtml+xml"
    )
    
    translated_book.add_item(comparison_item)
    
    # Set chapter order (just using first matched chapters for now)
    translated_book.toc = [comparison_item]
    translated_book.spine = [item for item in translated_book.get_items() 
                             if isinstance(item, epub.EpubItem)]
    
    epub.write_epub(output_path, translated_book)
    
    return output_path


def create_summary_table(book1_path: str, book2_path: str, output_path: str):
    """Create simple summary table comparing model statistics.
    
    Args:
        book1_path: First translation EPUB (e.g., nllb_light)
        book2_path: Second translation EPUB (e.g., mistral_heavy)  
        output_path: Output file path
    
    Returns:
        Output file path
    """
    # Read both books
    book1 = epub.read_epub(book1_path)
    book2 = epub.read_epub(book2_path)
    
    # Create summary EPUB
    book = epub.EpubBook()
    book.set_title("Translation Model Comparison Summary")
    book.set_description("Side-by-side comparison of different model outputs.")
    book.set_language('en')  # Summary in English
    
    # Simple CSS for table
    style = """
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
    """
    book.add_item(epub.EpubCss(style_content=style))
    
    from pathlib import Path
    
    # Create comparison table
    html = """
    <h1>Model Comparison Summary</h1>
    <table>
        <tr><th>Model</th><th>Strategy</th><th>Chapter Count</th><th>Pending Items</th></tr>
    """
    
    for b in [book1, book2]:
        chap_count = sum(1 for i in b.get_items() if isinstance(i, epub.EpubHtml))
        items_left_open = 0
        
        html += f"""
        <tr>
            <td>{os.path.basename(book1_path)}</td>
            <td>light</td>
            <td>{chap_count}</td>
            <td>{items_left_open}</td>
        </tr>
        """
    
    html += "</table>"
    html += "<p><i>Note: Full content comparison requires manual review via side-by-side EPUB.</i></p>"
    
    summary_item = epub.EpubItem(
        name="comparison.xhtml",
        file_name="comparison.xhtml",
        content=html,
        media_type="application/xhtml+xml"
    )
    
    book.add_item(summary_item)
    book.toc = [summary_item]
    book.spine = [summary_item]
    
    epub.write_epub(output_path, book)
    
    return output_path
