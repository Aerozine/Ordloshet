"""Auto-discover translator adapters in the translators/ package.

Any translators/translatorXxx.py that defines a module-level LABEL string is automatically
registered in the model registry under a normalized key derived from LABEL.

Key normalisation:  "NLLB-CT2" -> "nllb-ct2",  "Qwen-AWQ" -> "qwen-awq", etc.
Adapters that expose patch_translation() are also registered as patchers.

This runs at import time and is called by epub.py before building the argparse choices.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


_BASE_NAMES = frozenset({"base_seq2seq", "base_ct2", "base_llm", "base_llm_awq"})


def _label_to_key(label: str) -> str:
    return label.strip().lower().replace(" ", "-").replace("_", "-")


def discover_translators(
    existing_direct: dict[str, str],
    existing_patcher: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return updated (direct_modules, patcher_modules) including auto-discovered adapters."""
    direct = dict(existing_direct)
    patcher = dict(existing_patcher)

    translators_path = Path(__file__).parent.parent / "translators"
    pkg = "translators"

    for finder, module_name, _ in pkgutil.iter_modules([str(translators_path)]):
        if not module_name.startswith("translator"):
            continue
        stem = module_name[len("translator"):]
        if stem.lower() in _BASE_NAMES or stem.lower().startswith("ct2") or stem == "Optim":
            continue
        if module_name in {"translatorCT2Common", "translatorOptim"}:
            continue

        full_module = f"{pkg}.{module_name}"
        try:
            spec = importlib.util.find_spec(full_module)
            if spec is None:
                continue
            mod = importlib.import_module(full_module)
        except Exception:
            continue

        label = getattr(mod, "LABEL", None)
        if not label:
            # fall back to module-level _inst.LABEL
            inst = getattr(mod, "_inst", None)
            label = getattr(inst, "LABEL", None) if inst else None
        if not label:
            continue

        key = _label_to_key(label)
        if key not in direct:
            direct[key] = full_module

        if hasattr(mod, "patch_translation") and key not in patcher:
            patcher[key] = full_module

    return direct, patcher
