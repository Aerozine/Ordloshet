"""NLLB-200 3.3B seq2seq adapter."""
from .base_seq2seq import Seq2SeqBase


class _NLLB(Seq2SeqBase):
    MODEL_ID = "facebook/nllb-200-3.3B"
    LABEL = "NLLB"
    SRC_LANG = "eng_Latn"
    FORCED_BOS_LANG = "fra_Latn"
    MODEL_SIZE_MSG = "~6.5GB"


_inst = _NLLB()
ensure_model_loaded = _inst.ensure_model_loaded
large_translate = _inst.translate
translate_many = _inst.translate_many
count_tokens_many = _inst.count_tokens_many
unload_model = _inst.unload_model
translate = large_translate
largetranslate = large_translate
