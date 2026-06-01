"""pipeline/register.py  -  Register (tu/vous + formality) detector with override."""
from typing import Dict, List, Optional
import re


INFORMAL_WORDS = {
    'cheri', 'merde', 'putain', 'monpote', 'salut', 'tenas', 
    'tagueule', 'ya', 'cestcool', 'jemesuis', 'onsf'
}

FORMAL_WORDS = {
    'cheri', 'merci beaucoup', 'mesdames et messieurs', 
    'je vous prie', 'sivousplait', 'veuillez', 'monsieur', 'madame'
}

INFORMAL_PATTERNS = [r'\b(?:tu|ton|ta|tes)\b']
FORMAL_PATTERNS = [r'\b(?:vous|votre|vos)\b']


def _score_formality(text: str) -> float:
    """Compute formality score from -1 (very informal) to +1 (very formal)."""
    text_lower = text.lower()
    score = 0.0
    
    for word in set(text_lower.split()):
        if word in INFORMAL_WORDS:
            score -= 0.25
        elif word in FORMAL_WORDS:
            score += 0.25
    
    return max(-1.0, min(1.0, score))


def process_chapter(chapter_text: str, chapter_name: str = None, override: str = None) -> dict:
    """Process a chapter and detect its register level."""
    result = {
        'chapter': chapter_name or 'unknown',
        'detected_register': None,
        'formality_score': None,
        'override_applied': False,
        'override_value': None,
        'sample_phrases': [],
    }
    
    if override:
        register_map = {'informal': -0.8, 'formal': 0.8, 'mixed': 0.0}
        result['detected_register'] = override
        result['override_applied'] = True
        result['override_value'] = register_map.get(override, 0.0)
    else:
        score = _score_formality(chapter_text)
        result['formality_score'] = round(score, 3)
        
        if score < -0.5:
            result['detected_register'] = 'informal'
        elif score > 0.5:
            result['detected_register'] = 'formal'
        else:
            result['detected_register'] = 'mixed'
    
    sentences = chapter_text.split('.')[:10]
    for sent in sentences:
        if len(sent.strip()) > 3:
            sample = sent.strip()[:50].replace('.', '')
            result['sample_phrases'].append(sample)
            break
    
    return result
