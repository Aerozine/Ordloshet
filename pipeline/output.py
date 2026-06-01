"""pipeline/output.py  -  EPUB reconstruction after translation."""
import zipfile
from pathlib import Path


def pack_output(original_epub: str, translated_texts: list) -> Path:
    """Reconstruct EPUB with translated content."""
    
    source_epub = Path(original_epub)
    parent = source_epub.parent
    
    # Create output directories
    model_dir = Path("output") / "nllb" / "light"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = model_dir / f"translated-{source_epub.stem}.epub"
    
    print(f"   [PACK] Pack output to: {output_path}")
    
    # Create ZIP directly
    zip_buffer = zipfile.ZipFile(str(output_path), 'w', zipfile.ZIP_DEFLATED)
    
    try:
        for orig_name, _ in translated_texts:
            # Use str() to ensure we're passing a string
            trans_text = _
            
            html_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><meta charset="utf-8"/><title>{Path(orig_name).stem}</title></head>
<body><div>{trans_text}</div></body>
</html>"""
            
            zip_buffer.writestr(str(orig_name), html_content)
        
        print(f"   [OK] EPUB saved: {output_path}")
        return output_path
        
    finally:
        zip_buffer.close()


__all__ = ['pack_output']
