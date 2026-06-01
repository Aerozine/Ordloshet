"""MADLAD-400 3B MT adapter — faster, lower VRAM than the 10B variant."""
from .base_seq2seq import Seq2SeqBase


class _MADLAD3B(Seq2SeqBase):
    MODEL_ID = "google/madlad400-3b-mt"
    LABEL = "MADLAD-3B"
    SOURCE_PREFIX = "<2fr> "
    MODEL_SIZE_MSG = "~6GB"


_inst = _MADLAD3B()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
translate_many = _inst.translate_many
count_tokens_many = _inst.count_tokens_many
unload_model = _inst.unload_model
largetranslate = large_translate
