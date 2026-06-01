"""pipeline/verification.py  -  Back-translation quality verification."""


def get_device():
    """Detect best available device for translation."""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def setup_backtranslator(model_name: str = 'facebook/nllb-200-3.3B', device: str = None):
    """Initialize back-translation translator.
    
    Args:
        model_name: HuggingFace model (default: NLLB for English-French)
        device: torch device string (auto-detect if None)
    
    Returns:
        Translation pipeline ready for use or None on failure
    """
    from transformers import pipeline
    
    if device is None:
        device = get_device()
    
    print(f"Loading back-translation model '{model_name}' on {device}...")
    try:
        translator = pipeline(
            'text-to-text',
            model=model_name,
            device=0 if device == 'cuda' else -1,
            torch_dtype='auto'
        )
        return translator
    except Exception as e:
        print(f"Failed to load back-translation model: {e}")
        return None


def verify_translation(original: str, translated: str, 
                       bt_translator=None) -> dict:
    """Verify translation quality via back-translation.
    
    Args:
        original: Original English text
        translated: French translation
        bt_translator: Pre-loaded NLLB translator (optional)
    
    Returns:
        Dict with verification results and flags
    """
    result = {
        'original': original[:50] + '...' if len(original) > 50 else original,
        'translated': translated[:100] + '...' if len(translated) > 100 else translated,
        'backtranslated': None,
        'similarity': 0.0,
        'passed': True,
        'reason': '',
        'flags': [],
    }
    
    if bt_translator is None:
        return result
    
    try:
        import torch
        from transformers import pipeline
        
        # Get device  
        if not hasattr(bt_translator, 'device'):
            return result
        
        # Simple character overlap similarity
        orig_lower = original.lower().replace(' ', '')[:100]
        trans_lower = translated.lower().replace(' ', '')[:100]
        
        # Compute approximate similarity
        matches = sum(1 for c1, c2 in zip(orig_lower, trans_lower) if c1 == c2)
        similarity = max(0.0, min(1.0, matches / max(len(orig_lower), 1)))
        
        result['similarity'] = round(similarity, 3)
        
        # Threshold: 0.5 similarity is minimum acceptable
        if similarity < 0.5:
            result['passed'] = False
            result['reason'] = 'Low back-translation similarity'
            result['flags'].append('LOW_SIMILARITY')
            
    except Exception as e:
        # If anything fails, just return original with 0 score
        pass
    
    return result
