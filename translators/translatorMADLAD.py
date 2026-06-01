"""MADLAD-400 10B MT adapter."""
from .base_seq2seq import Seq2SeqBase


class _MADLAD(Seq2SeqBase):
    MODEL_ID = "google/madlad400-10b-mt"
    LABEL = "MADLAD"
    SOURCE_PREFIX = "<2fr> "
    MODEL_SIZE_MSG = "~6GB"


_inst = _MADLAD()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
translate_many = _inst.translate_many
count_tokens_many = _inst.count_tokens_many
unload_model = _inst.unload_model
largetranslate = large_translate
