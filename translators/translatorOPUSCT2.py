"""CTranslate2 OPUS-MT en-fr adapter."""
from pathlib import Path

from .base_ct2 import CT2TranslatorBase


class _OPUSCT2(CT2TranslatorBase):
    MODEL_ID = "Helsinki-NLP/opus-mt-en-fr"
    CT2_DIR = Path("models") / "ct2" / "opus-mt-en-fr"
    LABEL = "OPUS-CT2"


_inst = _OPUSCT2()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
translate_many = _inst.translate_many
translate_many_nbest = _inst.translate_many_nbest
count_tokens_many = _inst.count_tokens_many
unload_model = _inst.unload_model
largetranslate = large_translate
