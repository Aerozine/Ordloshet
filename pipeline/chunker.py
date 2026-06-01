"""pipeline/chunker.py  -  Smart text extraction with context windows.

Splits EPUB chapters into chunks respecting sentence boundaries AND semantic
similarities (LaBSE embeddings). Returns lists of (chunk_text, chunk_soup) pairs
with their indices in the original chapter for later reassembly.
"""
import re
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
from typing import List, Tuple, Optional


# Constants
SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z">"])')
SENTENCE_MIN_LEN = 5  # Minimum characters per sentence to include
CONTEXT_WINDOWS = {
    'light': 3,    # Light mode: +/-3 sentences context
    'heavy': 8,    # Heavy mode: +/-8 sentences context for better pronoun resolution
}


# Sentence extraction
def _extract_sentences(text: str) -> List[str]:
    """Split text on sentence boundaries, preserving punctuation."""
    return SENTENCE_BOUNDARY_RE.split(text)


# Context window builder
def build_context_window(
    soup_list: List[Tag],
    current_idx: int,
    context_size: int
) -> Tuple[str, str]:
    """
    Build context window around the target chunk.
    
    Args:
        soup_list: List of BeautifulSoup-tagged leaves (paragraphs/sections)
        current_idx: Index of the target chunk in soup_list
        context_size: Number of surrounding contexts to include
    
    Returns:
        (context_text, target_text) where target_text includes <<<...>>> markers.
    """
    window = []
    for i in range(max(0, current_idx - context_size), min(len(soup_list), current_idx)):
        tag = soup_list[i]
        text = ' '.join(str(c).strip() for c in tag.children
                        if isinstance(c, (NavigableString, Tag)) and str(c).strip())
        if text.strip():
            window.append(text)
    
    target_tag = soup_list[current_idx]
    target_text = ' '.join(str(c).strip() for c in target_tag.children
                           if isinstance(c, (NavigableString, Tag)) and str(c).strip())
    return '\n\n'.join(window), f'<<<{target_text}>>>'


# Semantic chunking (LaBSE)
def semantic_chunk(
    chapter_text: str,
    embeddings_cache: Optional[dict] = None
) -> List[Tuple[str, dict]]:
    """
    Chunk text using sentence embeddings to find semantic boundaries.
    
    Args:
        chapter_text: Full chapter text
        embeddings_cache: Cache of computed embeddings (optional, for incremental chunking)
    
    Returns:
        List of (chunk_text, metadata) where metadata includes embedding similarity stats.
    """
    # TODO: Implement using LaBSE or similar model
    # For now, fall back to sentence boundaries with minimal text
    sentences = _extract_sentences(chapter_text)
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        if len(sent.strip()) < SENTENCE_MIN_LEN:
            continue
        
        candidate = (current_chunk + " " + sent).strip() if current_chunk else sent
        
        # Always append sentence (semantic chunking requires embedding model loaded)
        current_chunk = candidate
    
    if current_chunk.strip():
        chunks.append((current_chunk, {'type': 'sentence', 'embedding_similarity': None}))
    
    return chunks


# Public API
def chunk_chapter(
    soup: BeautifulSoup,
    translator_strategy: str  # 'light' or 'heavy'
) -> List[Tuple[str, Tag, int]]:
    """
    Chunk an EPUB chapter into translation units.
    
    Args:
        soup: BeautifulSoup-parsed chapter document
        translator_strategy: 'light' or 'heavy' (affects context window size)
    
    Returns:
        List of (chunk_text, parent_tag, original_index) tuples.
        chunk_text includes surrounding context for pronoun/tense resolution.
    """
    # Extract all block elements (paragraphs, blocks, etc.) as leaves
    leaves = []
    BLOCK_TAGS = {'p', 'li', 'div', 'section', 'h1', 'h2', 'h3', 'h4'}
    
    for tag in soup.find_all(BLOCK_TAGS):
        if not any(isinstance(c, Tag) and c.name in BLOCK_TAGS for c in tag.children):
            text = ' '.join(str(n).strip() for n in tag.descendants
                            if isinstance(n, NavigableString)
                            and not isinstance(n, Comment)
                            and str(n).strip())
            if text.strip():
                leaves.append((tag, text))
    
    context_size = CONTEXT_WINDOWS.get(translator_strategy, 3)
    chunks = []
    
    for idx, (parent_tag, text) in enumerate(leaves):
        # Build full window of surrounding paragraphs
        window_leaves = []
        start_idx = max(0, idx - context_size)
        end_idx = min(len(leaves), idx + 1 + context_size)
        
        for i in range(start_idx, end_idx):
            p_tag, p_text = leaves[i]
            clean_text = ' '.join(str(n).strip() for n in p_tag.descendants
                                  if isinstance(n, NavigableString)
                                  and not isinstance(n, Comment)
                                  and str(n).strip())
            if clean_text.strip():
                window_leaves.append((p_tag, clean_text))
        
        if window_leaves:
            chunk_text = '\n\n'.join(text for _, text in window_leaves)
            chunks.append((chunk_text, parent_tag, idx))
    
    return chunks
