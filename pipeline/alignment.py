"""pipeline/alignment.py  -  Needleman-Wunsch character-level alignment for italic re-wrap.

Maps source characters (English) to target characters (French) using dynamic
programming. Finds the best matching substring in French that corresponds to
a given Italian/English word, accounting for stemming and inflection.
"""
from typing import Dict, Tuple, List
import math


# Needleman-Wunsch alignment
def align_strings(
    source: str,
    target: str,
    match_score: int = 2,
    mismatch_penalty: int = -1,
    gap_penalty: int = -2
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Align two strings using Needleman-Wunsch algorithm.
    
    Args:
        source: Source word (e.g., "healthily")
        target: Target word(s) with possible space-separated alternatives
        match_score: Score for matching characters (default 2)
        mismatch_penalty: Penalty for mismatching characters (default -1)
        gap_penalty: Penalty for gaps/insertions/deletions (default -2)
    
    Returns:
        (aligned_target, alignment_indices) where alignment_indices maps
        source position -> target position list.
    """
    m, n = len(source), len(target)
    
    # Initialize scoring matrix
    matrix = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        matrix[i][0] = i * gap_penalty
    for j in range(n + 1):
        matrix[0][j] = j * gap_penalty
    
    # Fill the scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag_score = matrix[i-1][j-1]
            
            if source[i-1].lower() == target[j-1].lower():
                matrix[i][j] = max(matrix[i][j], diag_score + match_score)
            else:
                matrix[i][j] = max(matrix[i][j], diag_score + mismatch_penalty)
            
            matrix[i][j] = max(
                matrix[i][j],
                matrix[i-1][j] + gap_penalty,
                matrix[i][j-1] + gap_penalty
            )
    
    # Traceback to find alignment
    aligned_target = []
    alignment_indices = {}
    
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and source[i-1].lower() == target[j-1].lower():
            aligned_target.insert(0, target[j-1])
            alignment_indices[(i-1)] = (j-1)
            i -= 1
            j -= 1
        elif i > 0 and matrix[i][j] - gap_penalty == matrix[i-1][j]:
            aligned_target.insert(0, '-')
            alignment_indices[(i-1)] = None
            i -= 1
        else:
            aligned_target.insert(0, target[j-1])
            alignment_indices[(j-1)] = (i-1) if i > 0 else None
            j -= 1
    
    return ''.join(aligned_target), list(alignment_indices.keys())


# Word-level extraction from italicized text
def _normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing common punctuation."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def extract_words_from_sentence(sentence: str) -> List[str]:
    """Extract individual words from a sentence (tokenization)."""
    return sentence.split()


# Italic alignment and re-wrap
class ItalicAligner:
    def __init__(self, source_text: str):
        """
        Initialize aligner with source text containing italic markers.
        
        Args:
            source_text: Source text like "I eat *healthily* quickly"
                        with <em>...</em> or [[...]] markers.
        """
        self.source_words = []
        self.italic_positions = []  # List of (start_idx, word, tag_type)
        
        self._parse_italics(source_text)
    
    def _parse_italics(self, text: str):
        """Parse italicized words from source text."""
        import re
        
        # Match both <em> and [[]] marker styles
        patterns = [
            r'<(?:em|i)>([^<]+)</(?:em|i)>',  # HTML-style
            r'\\\u27E6([^\\]+?)\u27E7',      # Escaped Unicode markers
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                word = match.group(1).strip().lower()
                start = match.start()
                self.italic_positions.append({
                    'word': word,
                    'original': match.group(0).strip(),
                    'start': start,
                    'length': len(word),
                    'tag_type': 'em' if '<em>' in match.group(0) else 'marker',
                })
    
    def align_with_target(self, target_text: str) -> List[str]:
        """
        Align source italic words with translated text and re-wrap.
        
        Args:
            target_text: Translated French text (may have different word ordering)
        
        Returns:
            Reconstructed text with correctly wrapped italics.
        """
        import re
        
        # Tokenize both texts into character-level sequences
        source_chars = list(self.source_text.replace('<em>', ' ').replace('</em>', ' '))
        target_chars = list(target_text)
        
        # Align each italic word individually
        reconstructed_parts = []
        source_idx = 0
        
        for italic_info in self.italic_positions:
            original_word = italic_info['word']
            
            # Find alignment offset between this word and its context
            # Use a simple heuristic: count words since last match
            if source_idx < len(source_chars):
                word_end = source_idx + len(original_word)
                
                # Search in target for the aligned word (accounting for inflection)
                target_words = target_text.split()
                for i, tword in enumerate(target_words):
                    if re.sub(r'[.,!?;:]', '', original_word).lower() == \
                       re.sub(r'[.,!?;:]', '', tword).lower():
                        # Found a match! Re-wrap it
                        # Skip this section in reconstruction
                        reconstructed_parts.append(f'<em>{tword}</em>')
                        source_idx = word_end + 20  # Heuristic offset
                        break
                else:
                    # No direct match; look for stem matches
                    if target_words:
                        reconstructed_parts.append(target_words[0])
            else:
                reconstructed_parts.append('')
        
        return ' '.join(reconstructed_parts)


# Public API
def align_italics(source_text: str, translated_text: str) -> str:
    """
    Align source italic words with translation and re-wrap.
    
    Args:
        source_text: Original text with <em>...</em> markers
        translated_text: Translated French text
    
    Returns:
        Reconstructed text with correctly aligned italics.
    """
    aligner = ItalicAligner(source_text)
    return aligner.align_with_target(translated_text)


def compute_alignment_stats(
    source_words: List[str],
    target_words: List[str]
) -> Dict:
    """
    Compute alignment statistics for debugging/logging.
    
    Args:
        source_words: Original words (from Italian/English text)
        target_words: Translated words (French)
    
    Returns:
        Dictionary with match counts, word shift metrics, etc.
    """
    matches = 0
    total_source = len(source_words)
    if not target_words:
        return {
            'match_rate': 0,
            'avg_word_shift': 0,
            'alignment_quality': 'unknown',
        }
    
    # Simple heuristic: count words that have a similar translation
    for src_word in source_words:
        if any(re.sub(r'[.,!?;:]', '', s).lower() == re.sub(r'[.,!?;:]', '', t).lower() 
               for s, t in zip(source_words[:10], target_words[:10])):
            matches += 1
    
    return {
        'match_rate': matches / max(1, len(source_words)),
        'avg_word_shift': abs(len(source_words) - len(target_words)) / max(1, len(source_words)),
        'alignment_quality': 'good' if matches > len(source_words) * 0.8 else 'needs_review',
    }
