"""pipeline/strategies/light.py  -  Single-pass baseline strategy.

Light mode provides a fast, single-pass translation without glossary, 
semantic chunking, or advanced rescoring. Suitable for quick prototyping
and establishing a quality baseline.
"""
from typing import Dict, Optional, List


class LightStrategy:
    """
    Single-pass translation strategy.
    
    Features:
    - Direct chunk translation without semantic boundaries
    - Standard beam search (8 beams)
    - No glossary or self-consistency
    - No post-processing alignment for seq2seq models
    
    Best for: Quick translations, model comparison baselines, smaller GPUs.
    """
    
    def __init__(self, translator, output_dir: Optional[str] = None):
        """
        Initialize light strategy.
        
        Args:
            translator: Translator instance (AbstractTranslator)
            output_dir: Base directory for outputs
        """
        self.translator = translator
        self.output_dir = output_dir or ''
        self._name = "light"
    
    @property
    def name(self) -> str:
        return self._name
    
    def get_beam_config(self) -> Dict:
        """Get beam search configuration for light mode."""
        return {
            'num_beams': 8,
            'length_penalty': 1.0,
            'early_stopping': True,
            'diverse_beam': False,
        }
    
    def translate(self, text: str) -> str:
        """
        Translate text using single-pass strategy.
        
        Args:
            text: Text segment to translate
    
        Returns:
            Translated text
        """
        return self.translator.translate(text)
    
    def process_chapter(self, chunks: List[tuple]) -> List[tuple]:
        """
        Process a list of (chunk_text, parent_tag, original_index) tuples.
        
        Args:
            chunks: List of chunk tuples from chunker.py
    
        Returns:
            List of translated chunks with same structure
        """
        translated_chunks = []
        for chunk_text, parent_tag, idx in chunks:
            translated = self.translate(chunk_text)
            translated_chunks.append((translated, parent_tag, idx))
        return translated_chunks
