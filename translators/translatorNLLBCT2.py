"""CTranslate2 NLLB-200 3.3B adapter."""
from pathlib import Path

from .base_ct2 import CT2TranslatorBase


class _NLLBCT2(CT2TranslatorBase):
    MODEL_ID = "facebook/nllb-200-3.3B"
    CT2_DIR = Path("models") / "ct2" / "nllb-200-3.3B"
    LABEL = "NLLB-CT2"
    SRC_LANG = "eng_Latn"
    TGT_LANG = "fra_Latn"


_inst = _NLLBCT2()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
translate_many = _inst.translate_many
translate_many_nbest = _inst.translate_many_nbest
count_tokens_many = _inst.count_tokens_many
unload_model = _inst.unload_model
largetranslate = large_translate
