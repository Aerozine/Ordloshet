"""CTranslate2 MADLAD-400 10B MT adapter."""
from pathlib import Path

from .base_ct2 import CT2TranslatorBase


class _MADLADCT2(CT2TranslatorBase):
    MODEL_ID = "google/madlad400-10b-mt"
    CT2_DIR = Path("models") / "ct2" / "madlad400-10b"
    LABEL = "MADLAD-CT2"
    SOURCE_PREFIX = "<2fr> "


_inst = _MADLADCT2()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
translate_many = _inst.translate_many
translate_many_nbest = _inst.translate_many_nbest
count_tokens_many = _inst.count_tokens_many
unload_model = _inst.unload_model
largetranslate = large_translate
