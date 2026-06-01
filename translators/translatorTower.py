"""TowerInstruct-7B LLM post-editor adapter."""
from .base_llm import LLMBase


class _Tower(LLMBase):
    MODEL_ID = "Unbabel/TowerInstruct-7B-v0.2"
    LABEL = "Tower"
    MODEL_SIZE_MSG = "~7B"


_inst = _Tower()
ensure_model_loaded = _inst.ensure_model_loaded
translate = _inst.translate
large_translate = _inst.translate
build_chapter_memo = _inst.build_chapter_memo
patch_translation = _inst.patch_translation
repair_translation = _inst.repair_translation
unload_model = _inst.unload_model
largetranslate = large_translate
