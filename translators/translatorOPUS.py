"""OPUS-MT en-fr adapter with fallback model IDs."""
from .base_seq2seq import Seq2SeqBase


class _OPUS(Seq2SeqBase):
    MODEL_IDS = [
        "Helsinki-NLP/opus-mt-en-fr",
        "Helsinki-NLP/opus-mt-tc-big-en-fr",
    ]
    MODEL_ID = "Helsinki-NLP/opus-mt-en-fr"
    LABEL = "OPUS"


_inst = _OPUS()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
translate_many = _inst.translate_many
count_tokens_many = _inst.count_tokens_many
unload_model = _inst.unload_model
largetranslate = large_translate
